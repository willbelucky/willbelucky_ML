# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 12.
"""
from zlib import crc32


def bytes_to_float(b):
    return float(crc32(b) & 0xffffffff) / 2 ** 32


def str_to_float(s):
    if type(s) is not str:
        return s
    return bytes_to_float(s.encode("utf-8"))
