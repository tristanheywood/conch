from ast import Dict
from itertools import permutations
import re
import struct
from collections import Counter
from functools import cached_property
from operator import itemgetter
from typing import Callable

import more_itertools


class PTXAnalyser:

    kernel_asm: Dict
    ptx_str: str

    def __init__(self, kernel_asm: Dict) -> None:
        self.kernel_asm = kernel_asm
        self.ptx_str = kernel_asm["ptx"]

    @staticmethod
    def FromKernel(kernel: Callable, **meta_params):

        cache_keys = list(kernel.cache[0].keys())
        filt_keys = list(
            filter(
                lambda key: key[2][:len(meta_params)] == tuple(
                    meta_params.values()), cache_keys))

        if len(filt_keys) > 1:
            print(f"Choosing arbitrary key from {len(filt_keys)}")

        key = filt_keys[-1]
        kernel_asm = kernel.cache[0][key].asm

        return PTXAnalyser(kernel_asm)

    @cached_property
    def classified_lines(self):
        clines = []

        for ln, line in enumerate(self.ptx_str.splitlines()):

            sline = line.strip()

            if ":" in line:
                tpe = "label"
            elif len(line) < 2 or line[0] != "\t" or line[
                    1] == "." or sline.startswith("//") or sline.startswith(
                        "{"):
                tpe = "other"
            else:
                tpe = "op"

            clines.append((ln, tpe, line))

        return clines

    @cached_property
    def split_ops(self):
        sops = []

        for ln, tpe, line in self.classified_lines:
            if tpe == "op":
                op, *args = line.split(None, 1)
                if op[:3] == "@%p":
                    # Some operations are prepended with @%p<num>, which means the operation
                    # is performed only on threads where this register holds a true value.
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-at
                    # So the op is the second word in the line.
                    op, *args = args[0].split(None, 1)
                if len(args) == 1:
                    args = tuple("".join(
                        args[0].removesuffix(";").split()).split(","))
                sops.append((op, args))

        return sops

    @cached_property
    def op_names(self):
        return list(map(itemgetter(0), self.split_ops))

    @cached_property
    def op_counts(self) -> Counter:
        return Counter(dict(Counter(self.op_names).most_common()))

    def get_float_constant_counts(self) -> Counter:
        hex_nums = re.findall(r"0f[0-9A-F]{8}", self.ptx_str)
        return Counter(dict(Counter(map(float_from_hex, hex_nums)).most_common()))

    def get_op_motifs(self, length: int, threshold: int = 5) -> Counter:
        ss_counter = Counter(more_itertools.windowed(self.op_names, length))
        motifs = Counter(
            {ss: c
             for ss, c in ss_counter.items() if c >= threshold})
        sorted_motifs = Counter(dict(motifs.most_common()))

        return sorted_motifs

        # For any groups of motifs which are just permutations of each other, keep the 
        # one with the highest count.
        unique_motifs = {}
        for motif, count in sorted_motifs.items():
            op_set = frozenset(motif)
            if not op_set in unique_motifs.keys():
                unique_motifs[op_set] = count
        return unique_motifs

        # For any groups of motifs which are just permutations of each other, keep the 
        # one with the highest count.
        unique_motifs = {}
        for motif, count in sorted_motifs.items():
            motif_perms = set(permutations(motif))
            if not any(unique_motifs in motif_perms
                       for unique_motifs in unique_motifs):
                unique_motifs[motif] = count

        return Counter(unique_motifs)

    def summarize_ptx(self):

        lines = []
        for ln, tpe, line in self.classified_lines:
            hex_nums = re.findall(r"0f[0-9A-F]{8}", line)

            comment = " " + ", ".join(f"{hex_num} = {float_from_hex(hex_num)}" for hex_num in hex_nums)
            lines.append(line + comment)

        return "\n".join(lines)


def float_from_hex(hex_str: str) -> float:
    return struct.unpack('!f', bytes.fromhex(hex_str.removeprefix("0f")))[0]
