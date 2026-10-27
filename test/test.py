# SPDX-License-Identifier: Apache-2.0
#
# cocotb tests for tt_um_sahpar_gemm4x4 -- 4x4 integer output-stationary
# systolic GEMM behind the 8-bit host byte protocol. Every result is checked
# against a Python reference multiply.

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

from gemm_protocol import (
    BUSY, CHIP_ID, DONE, K, N, OP_CLEAR_GEMM, OP_GEMM, OP_ID, OP_LOAD_A,
    OP_LOAD_B, OP_NOP, OP_STATUS, RX_READY, TX_VALID,
    load_matrix, matmul as tiled_matmul, matmul_ref, read_c, recv_byte,
    send_byte, wait_idle,
    OP_LOAD_A_REAL, OP_LOAD_B_REAL, gemm4_real, gemm4_real_ref,
    load_real_matrix, matmul_real, read_c_real, status_byte,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


def matmul(a, b):
    """a: N*K row-major, b: K*N row-major -> N*N row-major."""
    return [
        sum(a[i * K + k] * b[k * N + j] for k in range(K))
        for i in range(N)
        for j in range(N)
    ]


def fmt(m):
    return "\n".join(str(m[r * N:(r + 1) * N]) for r in range(N))


async def run_and_check(dut, a, b, label=""):
    await load_matrix(dut, OP_LOAD_A, a)
    await load_matrix(dut, OP_LOAD_B, b)
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    got = await read_c(dut)
    exp = matmul(a, b)
    assert got == exp, f"{label} mismatch\nA=\n{fmt(a)}\nB=\n{fmt(b)}\ngot=\n{fmt(got)}\nexp=\n{fmt(exp)}"
    return got


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_id(dut):
    """ID replies 0xA4 and the static uio direction bits are right."""
    await reset(dut)
    assert int(dut.uio_oe.value) == 0b0011_1010, f"uio_oe={int(dut.uio_oe.value):#010b}"
    assert int(dut.uio_out.value) & (1 << RX_READY), "RX_READY should be high"
    await send_byte(dut, OP_ID)
    byte = await recv_byte(dut)
    assert byte == CHIP_ID, f"expected ID {CHIP_ID:#x}, got {byte:#x}"


@cocotb.test()
async def test_status_idle(dut):
    await reset(dut)
    await send_byte(dut, OP_NOP)
    await send_byte(dut, OP_STATUS)
    assert (await recv_byte(dut)) & 0x1 == 0
    assert int(dut.uio_out.value) & (1 << BUSY) == 0
    assert int(dut.uio_out.value) & (1 << DONE) == 0, "DONE must be low after reset"


@cocotb.test()
async def test_gemm_counting(dut):
    """Small positive values, easy to eyeball in a waveform."""
    await reset(dut)
    a = list(range(1, 17))
    b = list(range(16, 0, -1))
    await run_and_check(dut, a, b, "counting")


@cocotb.test()
async def test_gemm_identity(dut):
    """A = I  ->  C = B (checks the skew lines up every row/column)."""
    await reset(dut)
    ident = [1 if i == j else 0 for i in range(N) for j in range(K)]
    b = [(-1) ** k * (k * 7 + 3) for k in range(K * N)]
    got = await run_and_check(dut, ident, b, "identity")
    assert got == b


@cocotb.test()
async def test_gemm_extremes(dut):
    """Corner magnitudes: accumulator must hold +65536 and -65024 exactly."""
    await reset(dut)
    await run_and_check(dut, [-128] * 16, [-128] * 16, "max-positive")
    await run_and_check(dut, [-128] * 16, [127] * 16, "max-negative")
    await run_and_check(dut, [127] * 16, [127] * 16, "127^2")


@cocotb.test()
async def test_gemm_random(dut):
    """Random signed operands, several back-to-back runs without reset."""
    await reset(dut)
    rng = random.Random(0x5A4)
    for n in range(6):
        a = [rng.randint(-128, 127) for _ in range(N * K)]
        b = [rng.randint(-128, 127) for _ in range(K * N)]
        await run_and_check(dut, a, b, f"random run {n}")


@cocotb.test()
async def test_reuse_operands(dut):
    """Only reload B; A must persist from the previous LOAD_A."""
    await reset(dut)
    a = [3, -1, 4, 1, -5, 9, 2, -6, 5, 3, -5, 8, 9, -7, 9, 3]
    b1 = [2, 7, -1, 8, 2, -8, 1, 8, -2, 8, 4, 5, 9, 0, -4, 5]
    b2 = [v * -2 for v in b1]
    await run_and_check(dut, a, b1, "first")
    await load_matrix(dut, OP_LOAD_B, b2)
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    assert await read_c(dut) == matmul(a, b2)


@cocotb.test()
async def test_busy_and_load_while_busy(dut):
    """BUSY pin rises during a run; a LOAD_A sent mid-run is swallowed whole
    (operands stay intact) and a CLEAR_GEMM mid-run is ignored."""
    await reset(dut)
    a = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8]
    b = [2, 0, 1, -1, 0, 3, 1, 1, 1, 1, -2, 0, -1, 2, 0, 4]
    exp = matmul(a, b)
    await load_matrix(dut, OP_LOAD_A, a)
    await load_matrix(dut, OP_LOAD_B, b)

    await send_byte(dut, OP_CLEAR_GEMM)
    # junk LOAD_A while busy: all 16 data bytes must be consumed and dropped
    await send_byte(dut, OP_LOAD_A)
    assert int(dut.uio_out.value) & (1 << BUSY), "BUSY pin should be high mid-run"
    for _ in range(16):
        await send_byte(dut, 0x7F)
    await wait_idle(dut)
    assert await read_c(dut) == exp, "LOAD_A during busy corrupted the result"

    # the stored A must still be the original: rerun and compare
    await send_byte(dut, OP_CLEAR_GEMM)
    await send_byte(dut, OP_CLEAR_GEMM)          # second one lands while busy -> ignored
    await wait_idle(dut)
    assert await read_c(dut) == exp, "LOAD_A during busy overwrote stored A"

    # protocol is still in sync: ID works
    await send_byte(dut, OP_ID)
    assert await recv_byte(dut) == CHIP_ID


@cocotb.test()
async def test_ena_low_ignores_input(dut):
    """While ena is low the decoder ignores the host byte stream."""
    await reset(dut)
    dut.ena.value = 0
    await send_byte(dut, OP_ID)
    for _ in range(10):
        await RisingEdge(dut.clk)
        assert int(dut.uio_out.value) & (1 << TX_VALID) == 0, "replied while ena=0"
    dut.ena.value = 1
    await send_byte(dut, OP_ID)
    assert await recv_byte(dut) == CHIP_ID


@cocotb.test()
async def test_done_flag(dut):
    """DONE (pin + STATUS bit 1) is sticky after a run and drops on the next start."""
    await reset(dut)
    await load_matrix(dut, OP_LOAD_A, list(range(16)))
    await load_matrix(dut, OP_LOAD_B, list(range(16)))
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    assert int(dut.uio_out.value) & (1 << DONE), "DONE pin should be high after a run"
    await send_byte(dut, OP_STATUS)
    assert await recv_byte(dut) == 0b10, "STATUS should be done=1, busy=0"
    # still set after unrelated traffic (sticky)
    await send_byte(dut, OP_ID)
    await recv_byte(dut)
    await load_matrix(dut, OP_LOAD_B, [1] * 16)
    assert int(dut.uio_out.value) & (1 << DONE), "DONE should survive LOAD/ID"
    # next run clears it while busy
    await send_byte(dut, OP_GEMM)
    await send_byte(dut, OP_NOP)
    v = int(dut.uio_out.value)
    assert v & (1 << BUSY) and not v & (1 << DONE), f"expected busy & !done mid-run, uio_out={v:#010b}"
    await wait_idle(dut)
    assert int(dut.uio_out.value) & (1 << DONE)


@cocotb.test()
async def test_gemm_accumulate(dut):
    """GEMM (0x20) adds onto C; CLEAR_GEMM (0x21) starts over."""
    await reset(dut)
    rng = random.Random(0xACC)
    a1, b1, a2, b2 = ([rng.randint(-128, 127) for _ in range(16)] for _ in range(4))
    await run_and_check(dut, a1, b1, "first clear run")
    await load_matrix(dut, OP_LOAD_A, a2)
    await load_matrix(dut, OP_LOAD_B, b2)
    await send_byte(dut, OP_GEMM)
    await wait_idle(dut)
    exp = [x + y for x, y in zip(matmul(a1, b1), matmul(a2, b2))]
    assert await read_c(dut) == exp, "C += A2*B2 mismatch"
    # CLEAR_GEMM must drop the history
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    assert await read_c(dut) == matmul(a2, b2), "CLEAR_GEMM did not clear"


@cocotb.test()
async def test_accumulate_limit(dut):
    """15 runs (1 CLEAR + 14 GEMM) of (-128)*(-128)*4 = 983,040 per element: the
    documented 21-bit limit (60 summed products), read back exactly."""
    await reset(dut)
    await load_matrix(dut, OP_LOAD_A, [-128] * 16)
    await load_matrix(dut, OP_LOAD_B, [-128] * 16)
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    for _ in range(14):
        await send_byte(dut, OP_GEMM)
        await wait_idle(dut)
    got = await read_c(dut)
    assert got == [15 * 4 * 16384] * 16, f"got {got[:4]}..."
    # most-negative corner of the same budget: 15 runs of (-128)*127*4
    await load_matrix(dut, OP_LOAD_B, [127] * 16)
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    for _ in range(14):
        await send_byte(dut, OP_GEMM)
        await wait_idle(dut)
    got = await read_c(dut)
    assert got == [-15 * 4 * 16256] * 16, f"got {got[:4]}..."


@cocotb.test()
async def test_tiled_8x8(dut):
    """Full 8x8 int8 matmul on the 4x4 chip via the shared host routine:
    4 output blocks, each = CLEAR_GEMM + GEMM over 2 K-blocks."""
    await reset(dut)
    rng = random.Random(0x88)
    A = [rng.randint(-128, 127) for _ in range(64)]
    B = [rng.randint(-128, 127) for _ in range(64)]
    assert await tiled_matmul(dut, A, B, 8, 8, 8) == matmul_ref(A, B, 8, 8, 8)


@cocotb.test()
async def test_tiled_ragged(dut):
    """Shapes that are not multiples of 4 (zero padding on the host): 5x7 @ 7x3."""
    await reset(dut)
    rng = random.Random(0x573)
    A = [rng.randint(-128, 127) for _ in range(5 * 7)]
    B = [rng.randint(-128, 127) for _ in range(7 * 3)]
    assert await tiled_matmul(dut, A, B, 5, 7, 3) == matmul_ref(A, B, 5, 7, 3)


@cocotb.test()
async def test_tiled_long_k(dut):
    """Inner dimension 64 = 16 K-blocks: more than the 15 runs the 21-bit
    accumulator allows, so the host splits it into two on-chip sums. All -128:
    each element is 64 * 16384 = 1,048,576 = 2^20, one past what 21 signed
    bits hold -- this only passes if the split really happens."""
    await reset(dut)
    A = [-128] * (4 * 64)
    B = [-128] * (64 * 4)
    assert await tiled_matmul(dut, A, B, 4, 64, 4) == matmul_ref(A, B, 4, 64, 4)


# ---------------------------------------------------------------------------
# real data (BF16 in, FP32 out) through the data preparation unit
# ---------------------------------------------------------------------------
def float_matmul(a, b, n, k, m):
    return [sum(a[i * k + t] * b[t * m + j] for t in range(k)) for i in range(n) for j in range(m)]


def assert_close(got, exp, tol, label=""):
    """Every element within tol * max|exp| of the true product."""
    ref = max(abs(v) for v in exp) or 1.0
    worst = max(abs(g - e) for g, e in zip(got, exp)) / ref
    assert worst <= tol, f"{label} worst relative error {worst:.4f} > {tol}\ngot={got}\nexp={exp}"


@cocotb.test()
async def test_real_exact(dut):
    """Values that fit 7 significant bits at one scale come out exact."""
    await reset(dut)
    a = [1.5, -2.0, 0.25, 3.0, -0.5, 1.0, 2.5, -3.75, 0.0, 1.25, -1.0, 2.0, 3.5, -0.75, 0.5, 1.75]
    b = [2.0, 0.5, -1.0, 1.5, -2.5, 3.0, 0.25, -0.5, 1.0, -1.25, 2.0, 0.75, -3.0, 1.0, -0.25, 2.5]
    got = await gemm4_real(dut, a, b)
    assert got == float_matmul(a, b, N, K, N)
    assert got == gemm4_real_ref(a, b)


@cocotb.test()
async def test_real_random(dut):
    """Random floats over several magnitudes: bit-exact against the model,
    and close to the true product."""
    await reset(dut)
    random.seed(11)
    for scale in (1e-3, 1.0, 37.0, 1e4):
        for _ in range(3):
            a = [random.uniform(-1, 1) * scale for _ in range(N * K)]
            b = [random.uniform(-1, 1) for _ in range(K * N)]
            got = await gemm4_real(dut, a, b)
            assert got == gemm4_real_ref(a, b), f"model mismatch at scale {scale}"
            assert_close(got, float_matmul(a, b, N, K, N), 0.05, f"scale {scale}")


@cocotb.test()
async def test_real_and_int_mixed(dut):
    """A as real data, B as int8 buffer data (scale 0)."""
    await reset(dut)
    a = [0.5 * (i - 7) for i in range(N * K)]
    b = [(3 * i) % 11 - 5 for i in range(K * N)]
    await load_real_matrix(dut, OP_LOAD_A_REAL, a)
    await load_matrix(dut, OP_LOAD_B, b)
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    assert await read_c_real(dut) == float_matmul(a, b, N, K, N)
    # the int8 path still reads back as integers
    await load_matrix(dut, OP_LOAD_A, [int(2 * v) for v in a])
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    assert await read_c(dut) == matmul([int(2 * v) for v in a], b)


@cocotb.test()
async def test_real_tiled(dut):
    """6x5 by 5x7 float matmul tiled onto the 4x4 array."""
    await reset(dut)
    random.seed(12)
    n, k, m = 6, 5, 7
    a = [random.uniform(-4, 4) for _ in range(n * k)]
    b = [random.uniform(-4, 4) for _ in range(k * m)]
    got = await matmul_real(dut, a, b, n, k, m)
    assert_close(got, float_matmul(a, b, n, k, m), 0.06, "tiled")


@cocotb.test()
async def test_scale_mismatch_flag(dut):
    """GEMM (accumulate) with a different scale sets STATUS bit 2;
    CLEAR_GEMM clears it."""
    await reset(dut)
    ones = [1.0] * (N * K)
    await load_real_matrix(dut, OP_LOAD_A_REAL, ones)
    await load_real_matrix(dut, OP_LOAD_B_REAL, ones)
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    await send_byte(dut, OP_GEMM)                     # same scale: fine
    await wait_idle(dut)
    assert not (await status_byte(dut)) & 4
    assert await read_c_real(dut) == [8.0] * (N * N)
    await load_real_matrix(dut, OP_LOAD_A_REAL, [4.0] * (N * K))
    await send_byte(dut, OP_GEMM)                     # different scale
    await wait_idle(dut)
    assert (await status_byte(dut)) & 4
    await send_byte(dut, OP_CLEAR_GEMM)
    await wait_idle(dut)
    assert not (await status_byte(dut)) & 4
    assert await read_c_real(dut) == [16.0] * (N * N)


@cocotb.test()
async def test_real_zeros_and_tiny(dut):
    """All-zero and widely spread inputs: zeros stay zero, tiny elements
    next to large ones flush to zero exactly as modelled."""
    await reset(dut)
    a = [0.0] * (N * K)
    b = [1.0] * (K * N)
    assert await gemm4_real(dut, a, b) == [0.0] * (N * N)
    a = [1000.0 if i % 5 == 0 else 1e-3 for i in range(N * K)]
    got = await gemm4_real(dut, a, b)
    assert got == gemm4_real_ref(a, b)
    # results below the smallest FP32 normal (2^-126) flush to zero:
    # 2^-64 * 1.5 * 2^-63 = 1.5 * 2^-127
    a = [2.0 ** -64 if i % 5 == 0 else 0.0 for i in range(N * K)]
    b = [1.5 * 2.0 ** -63 if i % 5 == 0 else 0.0 for i in range(K * N)]
    assert await gemm4_real(dut, a, b) == [0.0] * (N * N)
    # just above it stays exact: 1.5 * 2^-126
    b = [1.5 * 2.0 ** -62 if i % 5 == 0 else 0.0 for i in range(K * N)]
    assert await gemm4_real(dut, a, b) == [1.5 * 2.0 ** -126 if i % 5 == 0 else 0.0 for i in range(N * N)]
