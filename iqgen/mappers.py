"""Constellation mappers for all supported modulations.

Bit-packing convention: within each symbol, bits are read MSB-first.
For QPSK, bits "01" -> index 1, bits "10" -> index 2, etc.

Gray mapping is applied by transforming the integer index via i -> i ^ (i>>1)
before computing the constellation phase, which puts adjacent constellation
points one bit apart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np


SymbolOutput = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]


def bits_to_indices(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Pack bits MSB-first within each symbol group."""
    if bits.size == 0:
        return np.zeros(0, dtype=np.int64)
    groups = bits.reshape(-1, bits_per_symbol).astype(np.int64)
    weights = 1 << np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int64)
    return groups @ weights


def binary_to_gray(n: np.ndarray) -> np.ndarray:
    return n ^ (n >> 1)


class ModulationMapper(ABC):
    bits_per_symbol: int = 0

    @abstractmethod
    def map(self, bits: np.ndarray) -> SymbolOutput:
        ...


class BPSKMapper(ModulationMapper):
    bits_per_symbol = 1

    def map(self, bits):
        # 0 -> 1+0j, 1 -> -1+0j (Gray-trivial)
        return np.where(bits == 0, 1.0 + 0.0j, -1.0 + 0.0j).astype(np.complex64)


class QPSKMapper(ModulationMapper):
    """Gray-coded QPSK per spec: 00->0°, 01->90°, 11->180°, 10->270°."""
    bits_per_symbol = 2

    def __init__(self, gray_coding: bool = True):
        self.gray_coding = gray_coding

    def map(self, bits):
        idx = bits_to_indices(bits, 2)
        mapping = binary_to_gray(idx) if self.gray_coding else idx
        phase = mapping * (np.pi / 2)
        return np.exp(1j * phase).astype(np.complex64)


class PSK8Mapper(ModulationMapper):
    """Gray-coded 8PSK: 8 points equally spaced on unit circle."""
    bits_per_symbol = 3

    def __init__(self, gray_coding: bool = True):
        self.gray_coding = gray_coding

    def map(self, bits):
        idx = bits_to_indices(bits, 3)
        mapping = binary_to_gray(idx) if self.gray_coding else idx
        phase = mapping * (2 * np.pi / 8)
        return np.exp(1j * phase).astype(np.complex64)


class DifferentialMapper(ModulationMapper):
    """Differential wrapper: each base-mapped symbol is treated as a phase
    rotation applied to the previous output symbol. Initial phase configurable.
    """

    def __init__(self, base: ModulationMapper, initial_phase: float = 0.0):
        self.base = base
        self.bits_per_symbol = base.bits_per_symbol
        self.initial_phase = initial_phase

    def map(self, bits):
        deltas = self.base.map(bits)
        if isinstance(deltas, tuple):
            raise TypeError("Cannot wrap a tuple-output mapper (e.g. OQPSK) differentially")
        if deltas.size == 0:
            return deltas
        # Cumulative complex product
        cum = np.cumprod(deltas)
        return (np.exp(1j * self.initial_phase) * cum).astype(np.complex64)


class Pi4QPSKMapper(ModulationMapper):
    """π/4-DQPSK per IS-54 phase-delta table (Gray-coded).

    bits -> Δφ:
        00 ->  +π/4
        01 -> +3π/4
        10 ->  -π/4
        11 -> -3π/4
    Symbols alternate between two QPSK constellations offset by π/4.
    """
    bits_per_symbol = 2

    # Index = bits packed MSB-first ([msb,lsb] -> msb*2+lsb)
    DELTA_TABLE = np.array([np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4])

    def __init__(self, gray_coding: bool = True, initial_phase: float = 0.0):
        # gray_coding is implicit in the delta table; flag kept for API symmetry
        self.gray_coding = gray_coding
        self.initial_phase = initial_phase

    def map(self, bits):
        idx = bits_to_indices(bits, 2)
        if idx.size == 0:
            return np.zeros(0, dtype=np.complex64)
        deltas = self.DELTA_TABLE[idx]
        phases = np.cumsum(deltas) + self.initial_phase
        return np.exp(1j * phases).astype(np.complex64)


class Pi4PSK8Mapper(ModulationMapper):
    """π/4-8PSK: 8PSK with every other symbol rotated by π/8.
    Eliminates zero-crossings through the origin.
    """
    bits_per_symbol = 3

    def __init__(self, gray_coding: bool = True):
        self.gray_coding = gray_coding
        self._base = PSK8Mapper(gray_coding=gray_coding)

    def map(self, bits):
        symbols = self._base.map(bits)
        if symbols.size == 0:
            return symbols
        n = np.arange(symbols.size)
        rotation = np.exp(1j * (np.pi / 8) * (n % 2))
        return (symbols * rotation).astype(np.complex64)


class Pi2BPSKMapper(ModulationMapper):
    """π/2-BPSK (used in 5G NR uplink low-PAPR mode). Each successive symbol
    is rotated by an additional π/2, so even-indexed symbols sit on the real
    axis (±1) and odd-indexed on the imaginary axis (±j). This eliminates
    180° transitions and roughly halves the PAPR of plain BPSK.
    """
    bits_per_symbol = 1

    def map(self, bits):
        base = np.where(bits == 0, 1.0 + 0.0j, -1.0 + 0.0j)
        if base.size == 0:
            return base.astype(np.complex64)
        n = np.arange(base.size)
        rotation = np.exp(1j * (np.pi / 2) * n)
        return (base * rotation).astype(np.complex64)


class OQPSKMapper(ModulationMapper):
    """Offset QPSK. Returns (I, Q) tuple so the generator can delay Q by half
    a symbol after upsampling. Uses the same Gray-coded constellation as QPSK.
    """
    bits_per_symbol = 2

    def __init__(self, gray_coding: bool = True):
        self._qpsk = QPSKMapper(gray_coding=gray_coding)

    def map(self, bits):
        symbols = self._qpsk.map(bits)
        return (
            symbols.real.astype(np.float32),
            symbols.imag.astype(np.float32),
        )


def create_mapper(modulation: str, gray_coding: bool = True,
                  initial_phase: float = 0.0) -> ModulationMapper:
    m = modulation.lower()
    if m == "bpsk":
        return BPSKMapper()
    if m == "pi2_bpsk":
        return Pi2BPSKMapper()
    if m == "qpsk":
        return QPSKMapper(gray_coding=gray_coding)
    if m == "8psk":
        return PSK8Mapper(gray_coding=gray_coding)
    if m == "dbpsk":
        return DifferentialMapper(BPSKMapper(), initial_phase=initial_phase)
    if m == "dqpsk":
        return DifferentialMapper(QPSKMapper(gray_coding=gray_coding),
                                   initial_phase=initial_phase)
    if m == "d8psk":
        return DifferentialMapper(PSK8Mapper(gray_coding=gray_coding),
                                   initial_phase=initial_phase)
    if m == "pi4_qpsk":
        return Pi4QPSKMapper(gray_coding=gray_coding, initial_phase=initial_phase)
    if m == "pi4_8psk":
        return Pi4PSK8Mapper(gray_coding=gray_coding)
    if m == "oqpsk":
        return OQPSKMapper(gray_coding=gray_coding)
    raise ValueError(f"Unknown modulation: {modulation}")
