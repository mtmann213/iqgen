"""Packet framing for iqgen.

Builds a framed bit stream from a payload, and parses a (possibly noisy)
bit stream back into a payload plus a diagnostic report.

On-wire layout:
    [preamble][syncword][FEC( header || payload || CRC )]

Preamble and syncword are sent uncoded (they're used to locate the frame
on the receive side). Everything from the header through the CRC is
covered by the chosen FEC. CRC is computed over (header || payload)
*before* FEC.

All bit arrays are MSB-first numpy uint8 0/1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =============================================================================
# Bit helpers
# =============================================================================

def bytes_to_bits(b: bytes) -> np.ndarray:
    """MSB-first unpack."""
    if not b:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="big")


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """MSB-first pack; bits length must be a multiple of 8."""
    if bits.size == 0:
        return b""
    return np.packbits(bits.astype(np.uint8), bitorder="big").tobytes()


def int_to_bits(value: int, n_bits: int) -> np.ndarray:
    """MSB-first."""
    out = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        out[n_bits - 1 - i] = (value >> i) & 1
    return out


def bits_to_int(bits: np.ndarray) -> int:
    """MSB-first."""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


# =============================================================================
# CRC
# =============================================================================

CRC_SPECS = {
    # name: (width, poly, init, refin, refout, xorout)
    "crc16-ccitt-false": (16, 0x1021, 0xFFFF, False, False, 0x0000),
    "crc32":             (32, 0x04C11DB7, 0xFFFFFFFF, True, True, 0xFFFFFFFF),
}


def _crc_bits_to_int(bits: np.ndarray, width: int, poly: int, init: int,
                     refin: bool, refout: bool, xorout: int) -> int:
    """Compute CRC over a bit-array (MSB-first)."""
    # Convert bit-array back to bytes for the standard byte-wise CRC algo.
    if bits.size % 8 != 0:
        pad = np.zeros(8 - (bits.size % 8), dtype=np.uint8)
        bits = np.concatenate([bits, pad])
    data = bits_to_bytes(bits)
    reg = init
    top = 1 << (width - 1)
    mask = (1 << width) - 1
    for byte in data:
        if refin:
            byte = int(f"{byte:08b}"[::-1], 2)
        reg ^= byte << (width - 8)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    if refout:
        reg = int(f"{reg:0{width}b}"[::-1], 2)
    return reg ^ xorout


def crc_compute(bits: np.ndarray, scheme: str) -> tuple[int, int]:
    """Return (crc_value, crc_width_in_bits)."""
    if scheme == "none":
        return 0, 0
    spec = CRC_SPECS.get(scheme)
    if spec is None:
        raise ValueError(f"Unknown CRC scheme: {scheme}")
    width, poly, init, refin, refout, xorout = spec
    return _crc_bits_to_int(bits, width, poly, init, refin, refout, xorout), width


def crc_width(scheme: str) -> int:
    if scheme == "none":
        return 0
    return CRC_SPECS[scheme][0]


# =============================================================================
# FEC
# =============================================================================
# Hamming(7,4) - systematic. Data bits d0..d3, parity p0..p2.
# Codeword layout (MSB-first within the 7-bit block):
#   c = [d0, d1, d2, d3, p0, p1, p2]
# Parity equations (over GF(2)):
#   p0 = d0 ^ d1 ^ d2
#   p1 = d1 ^ d2 ^ d3
#   p2 = d0 ^ d1 ^ d3
# Syndrome bits s = (s0, s1, s2):
#   s0 = c0 ^ c1 ^ c2 ^ c4
#   s1 = c1 ^ c2 ^ c3 ^ c5
#   s2 = c0 ^ c1 ^ c3 ^ c6
# Syndrome → error position lookup gives single-bit error correction.

_HAMMING_SYNDROME_LUT = {
    (0, 0, 0): None,   # no error
    (1, 0, 1): 0,      # d0 bit
    (1, 1, 1): 1,      # d1 bit
    (1, 1, 0): 2,      # d2 bit
    (0, 1, 1): 3,      # d3 bit
    (1, 0, 0): 4,      # p0 bit
    (0, 1, 0): 5,      # p1 bit
    (0, 0, 1): 6,      # p2 bit
}


def _hamming74_encode_block(d: np.ndarray) -> np.ndarray:
    """Encode 4 data bits -> 7 codeword bits."""
    d0, d1, d2, d3 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
    p0 = d0 ^ d1 ^ d2
    p1 = d1 ^ d2 ^ d3
    p2 = d0 ^ d1 ^ d3
    return np.array([d0, d1, d2, d3, p0, p1, p2], dtype=np.uint8)


def _hamming74_decode_block(c: np.ndarray) -> tuple[np.ndarray, Optional[int]]:
    """Decode 7 received bits -> (4 data bits, error_position or None)."""
    c0, c1, c2, c3, c4, c5, c6 = (int(c[i]) for i in range(7))
    s0 = c0 ^ c1 ^ c2 ^ c4
    s1 = c1 ^ c2 ^ c3 ^ c5
    s2 = c0 ^ c1 ^ c3 ^ c6
    err_pos = _HAMMING_SYNDROME_LUT.get((s0, s1, s2))
    corrected = c.copy()
    if err_pos is not None:
        corrected[err_pos] ^= 1
    return corrected[:4], err_pos


def fec_encode(bits: np.ndarray, scheme: str) -> np.ndarray:
    """Encode a bit stream with the chosen FEC. The data is zero-padded
    up to the codeword boundary; the receiver must know the original
    data length to strip the pad (or accept the trailing pad bits)."""
    if scheme == "none":
        return bits.astype(np.uint8)
    if scheme == "repetition-3":
        return np.repeat(bits.astype(np.uint8), 3)
    if scheme == "hamming-7-4":
        n = bits.size
        pad = (-n) % 4
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        blocks = bits.reshape(-1, 4)
        out = np.zeros(blocks.shape[0] * 7, dtype=np.uint8)
        for i, b in enumerate(blocks):
            out[i * 7:(i + 1) * 7] = _hamming74_encode_block(b)
        return out
    raise ValueError(f"Unknown FEC scheme: {scheme}")


@dataclass
class FecDecodeResult:
    bits: np.ndarray                 # decoded data bits (with any tail pad)
    n_codewords: int
    n_corrections: int               # total codewords that needed a correction
    n_uncorrectable: int             # codewords that exceeded correction capability (informational; rep-3 can't really detect this)
    corrected_positions: list[int]   # GLOBAL bit positions (in the decoded output) that were flipped


def fec_decode(bits: np.ndarray, scheme: str,
               expected_data_bits: Optional[int] = None) -> FecDecodeResult:
    """Decode a (possibly noisy) coded stream back to data bits. If
    `expected_data_bits` is given, the output is trimmed to that length.
    `corrected_positions` lists DATA-bit positions (post-decode) that
    differ from a straight majority-vote (rep-3) or were syndrome-corrected
    (Hamming) — these are the diagnostic 'FEC saved us' indicators."""
    if scheme == "none":
        out = bits.astype(np.uint8).copy()
        if expected_data_bits is not None:
            out = out[:expected_data_bits]
        return FecDecodeResult(out, 0, 0, 0, [])
    if scheme == "repetition-3":
        n = bits.size
        usable = (n // 3) * 3
        triples = bits[:usable].reshape(-1, 3)
        majority = (triples.sum(axis=1) >= 2).astype(np.uint8)
        # A correction event = any codeword where the bits weren't unanimous.
        unanimous = (triples.sum(axis=1) % 3 == 0)  # all-0 or all-1
        corrected_idx = np.where(~unanimous)[0].tolist()
        out = majority
        if expected_data_bits is not None:
            out = out[:expected_data_bits]
            corrected_idx = [i for i in corrected_idx if i < expected_data_bits]
        return FecDecodeResult(out, triples.shape[0], len(corrected_idx), 0,
                                corrected_idx)
    if scheme == "hamming-7-4":
        n = bits.size
        usable = (n // 7) * 7
        blocks = bits[:usable].reshape(-1, 7)
        n_blocks = blocks.shape[0]
        out = np.zeros(n_blocks * 4, dtype=np.uint8)
        corrections: list[int] = []
        for i in range(n_blocks):
            data, err = _hamming74_decode_block(blocks[i])
            out[i * 4:(i + 1) * 4] = data
            if err is not None:
                # report the position of the corrected DATA bit if the error
                # was in the data portion; if it was in a parity bit, we
                # still flag the codeword (use the data-bit base index).
                if err < 4:
                    corrections.append(i * 4 + err)
                else:
                    corrections.append(i * 4)  # parity-bit correction; flag codeword
        if expected_data_bits is not None:
            out = out[:expected_data_bits]
            corrections = [i for i in corrections if i < expected_data_bits]
        return FecDecodeResult(out, n_blocks, len(corrections), 0, corrections)
    raise ValueError(f"Unknown FEC scheme: {scheme}")


def fec_overhead_bits(payload_bits: int, scheme: str) -> int:
    """Number of encoded bits produced for `payload_bits` of data."""
    if scheme == "none":
        return payload_bits
    if scheme == "repetition-3":
        return payload_bits * 3
    if scheme == "hamming-7-4":
        return ((payload_bits + 3) // 4) * 7
    raise ValueError(f"Unknown FEC scheme: {scheme}")


# =============================================================================
# Sync correlation
# =============================================================================

def find_sync(bits: np.ndarray, pattern: np.ndarray,
              max_distance: Optional[int] = None) -> tuple[int, int]:
    """Slide `pattern` across `bits`, return (best_offset, min_distance).
    Distance is Hamming distance (count of differing bits). If
    `max_distance` is set, returns the first position with distance ≤
    that; otherwise the global minimum.
    """
    if pattern.size == 0:
        return 0, 0
    if bits.size < pattern.size:
        return -1, pattern.size
    pat = pattern.astype(np.uint8)
    n = bits.size - pat.size + 1
    # Vectorized: cumulative XOR comparison. For modest pattern sizes,
    # straightforward loop is fine and clear.
    best_off, best_dist = -1, pat.size + 1
    for i in range(n):
        d = int(np.sum(bits[i:i + pat.size] != pat))
        if d < best_dist:
            best_dist = d
            best_off = i
            if d == 0:
                break
        if max_distance is not None and d <= max_distance:
            return i, d
    return best_off, best_dist


# =============================================================================
# Frame config
# =============================================================================

# Default preset values
_DEFAULT_PREAMBLE = bytes([0xAA] * 4)
_DEFAULT_SYNCWORD = bytes.fromhex("1ACFFC1D")
_DEFAULT_HEADER_FORMAT = (("length", 16), ("seq", 8), ("type", 8))


@dataclass
class FrameConfig:
    preamble: bytes = _DEFAULT_PREAMBLE
    syncword: bytes = _DEFAULT_SYNCWORD
    header_format: tuple = _DEFAULT_HEADER_FORMAT  # ordered (name, n_bits)
    crc: str = "crc16-ccitt-false"   # "none" | "crc16-ccitt-false" | "crc32"
    fec: str = "hamming-7-4"          # "none" | "repetition-3" | "hamming-7-4"

    def header_bits_total(self) -> int:
        return sum(b for _, b in self.header_format)

    def field_names(self) -> list[str]:
        return [n for n, _ in self.header_format]

    def field_width(self, name: str) -> int:
        for n, b in self.header_format:
            if n == name:
                return b
        raise KeyError(name)


# =============================================================================
# Frame build
# =============================================================================

def _pack_header(header_values: dict, fmt: tuple) -> np.ndarray:
    parts = []
    for name, width in fmt:
        v = int(header_values.get(name, 0))
        if v >= (1 << width) or v < 0:
            raise ValueError(f"Header field {name!r}={v} doesn't fit in {width} bits")
        parts.append(int_to_bits(v, width))
    if not parts:
        return np.zeros(0, dtype=np.uint8)
    return np.concatenate(parts)


def build_frame(payload_bits: np.ndarray,
                cfg: FrameConfig,
                header_values: Optional[dict] = None) -> np.ndarray:
    """Build a full framed bit stream from a payload.

    If `header_values` omits 'length', it is auto-filled with the payload
    length in bits.
    """
    payload_bits = np.asarray(payload_bits, dtype=np.uint8)
    hv = dict(header_values or {})
    if "length" in cfg.field_names() and "length" not in hv:
        hv["length"] = payload_bits.size

    header_bits = _pack_header(hv, cfg.header_format)

    # CRC is computed over (header || payload) before FEC.
    crc_input = np.concatenate([header_bits, payload_bits])
    crc_val, crc_w = crc_compute(crc_input, cfg.crc)
    crc_bits = int_to_bits(crc_val, crc_w) if crc_w else np.zeros(0, dtype=np.uint8)

    # FEC over the whole protected block.
    protected = np.concatenate([header_bits, payload_bits, crc_bits])
    coded = fec_encode(protected, cfg.fec)

    preamble = bytes_to_bits(cfg.preamble)
    sync = bytes_to_bits(cfg.syncword)
    return np.concatenate([preamble, sync, coded]).astype(np.uint8)


# =============================================================================
# Frame parse
# =============================================================================

@dataclass
class FrameReport:
    # Sync
    preamble_offset: int = -1
    sync_offset: int = -1
    sync_distance: int = -1
    sync_pattern_bits: int = 0
    sync_found: bool = False

    # Header
    header_bits: Optional[np.ndarray] = None
    header_fields: dict = field(default_factory=dict)

    # FEC
    fec_scheme: str = "none"
    fec_codewords: int = 0
    fec_corrections: int = 0
    fec_corrected_positions: list = field(default_factory=list)

    # CRC
    crc_scheme: str = "none"
    crc_expected: Optional[int] = None
    crc_computed: Optional[int] = None
    crc_ok: bool = False

    # Payload
    payload_bits: Optional[np.ndarray] = None
    payload_length_declared: int = 0   # from header.length

    # Comparison vs expected (optional)
    n_payload_errors: Optional[int] = None
    n_payload_compared: int = 0

    notes: list = field(default_factory=list)

    def short_summary(self) -> str:
        bits = "yes" if self.sync_found else "NO"
        crc = "PASS" if self.crc_ok else ("FAIL" if self.crc_scheme != "none" else "n/a")
        return (f"sync={bits} (dist {self.sync_distance}/"
                f"{self.sync_pattern_bits})  fec_corr={self.fec_corrections}  "
                f"crc={crc}  payload={0 if self.payload_bits is None else self.payload_bits.size} bits")


def _unpack_header(bits: np.ndarray, fmt: tuple) -> dict:
    out, pos = {}, 0
    for name, width in fmt:
        out[name] = bits_to_int(bits[pos:pos + width])
        pos += width
    return out


def parse_frame(bits: np.ndarray, cfg: FrameConfig,
                expected_payload_bits: Optional[np.ndarray] = None,
                sync_search_window: Optional[int] = None,
                max_sync_distance: Optional[int] = None) -> FrameReport:
    """Locate and decode one frame in `bits`.

    `sync_search_window` limits how far into `bits` we hunt for the sync
    pattern; default is the entire stream.
    `max_sync_distance` lets you tolerate noisy sync words — if set, the
    first position with distance ≤ this threshold wins (faster, more
    permissive).
    """
    bits = np.asarray(bits, dtype=np.uint8)
    rep = FrameReport(fec_scheme=cfg.fec, crc_scheme=cfg.crc)

    # 1. Find preamble + syncword as one long pattern.
    preamble = bytes_to_bits(cfg.preamble)
    sync = bytes_to_bits(cfg.syncword)
    full_pattern = np.concatenate([preamble, sync])
    rep.sync_pattern_bits = full_pattern.size

    search_bits = bits if sync_search_window is None else bits[:sync_search_window + full_pattern.size]
    off, dist = find_sync(search_bits, full_pattern, max_sync_distance)
    if off < 0:
        rep.notes.append("sync not found within search window")
        return rep
    rep.preamble_offset = off
    rep.sync_offset = off + preamble.size
    rep.sync_distance = dist
    # Heuristic: accept if distance ≤ 1/4 of pattern bits (very permissive
    # for the noise-free path; tighten for noisy channels later).
    threshold = max(1, full_pattern.size // 4)
    rep.sync_found = dist <= threshold
    if not rep.sync_found:
        rep.notes.append(
            f"sync distance {dist} > threshold {threshold}; aborting decode")
        return rep

    # 2. Slice the coded portion (header || payload || crc, FEC-encoded).
    coded_start = off + full_pattern.size
    coded = bits[coded_start:]

    # 3. We don't yet know the payload length, so decode FEC over what we
    #    have, then read the header to find the declared payload length.
    header_bits_n = cfg.header_bits_total()
    crc_w = crc_width(cfg.crc)

    # First decode enough to read the header.
    header_coded_n = fec_overhead_bits(header_bits_n, cfg.fec)
    if coded.size < header_coded_n:
        rep.notes.append("truncated: not enough coded bits to read header")
        return rep
    hdr_dec = fec_decode(coded[:header_coded_n], cfg.fec,
                         expected_data_bits=header_bits_n)
    header_bits = hdr_dec.bits
    rep.header_bits = header_bits
    rep.header_fields = _unpack_header(header_bits, cfg.header_format)
    payload_n = int(rep.header_fields.get("length", 0))
    rep.payload_length_declared = payload_n

    # 4. Now decode the full protected block (header + payload + crc).
    protected_n = header_bits_n + payload_n + crc_w
    coded_n = fec_overhead_bits(protected_n, cfg.fec)
    if coded.size < coded_n:
        rep.notes.append("truncated: declared payload length exceeds available bits")
        coded_n = coded.size
        protected_n = min(protected_n,
                          {"none": coded_n,
                           "repetition-3": coded_n // 3,
                           "hamming-7-4": (coded_n // 7) * 4}[cfg.fec])

    full_dec = fec_decode(coded[:coded_n], cfg.fec,
                          expected_data_bits=protected_n)
    rep.fec_codewords = full_dec.n_codewords
    rep.fec_corrections = full_dec.n_corrections
    rep.fec_corrected_positions = full_dec.corrected_positions
    protected = full_dec.bits

    # Re-extract header (now from the consistent full decode) just to keep
    # everything coming from the same decode.
    rep.header_bits = protected[:header_bits_n]
    rep.header_fields = _unpack_header(rep.header_bits, cfg.header_format)
    payload = protected[header_bits_n:header_bits_n + payload_n]
    rep.payload_bits = payload

    # 5. CRC check.
    if cfg.crc != "none":
        crc_bits_received = protected[header_bits_n + payload_n:
                                      header_bits_n + payload_n + crc_w]
        rep.crc_expected = bits_to_int(crc_bits_received)
        crc_val, _ = crc_compute(
            np.concatenate([rep.header_bits, payload]), cfg.crc)
        rep.crc_computed = crc_val
        rep.crc_ok = (rep.crc_expected == rep.crc_computed)
    else:
        rep.crc_ok = True  # no CRC = no failure mode

    # 6. Optional payload comparison.
    if expected_payload_bits is not None and payload is not None:
        n = min(payload.size, expected_payload_bits.size)
        rep.n_payload_compared = n
        rep.n_payload_errors = int(np.sum(
            payload[:n] != expected_payload_bits[:n].astype(np.uint8)))

    return rep
