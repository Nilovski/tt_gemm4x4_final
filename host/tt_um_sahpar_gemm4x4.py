# SPDX-License-Identifier: Apache-2.0
#
# Hardware-in-the-loop runner for tt_um_sahpar_gemm4x4 on the Tiny Tapeout
# demo board (RP2040/RP2350 + tt-micropython-firmware, which ships microcotb).
#
# It reuses ../test/gemm_protocol.py -- the exact host code the cocotb
# simulation tests exercise -- so what passes in sim is what drives the chip.
#
# Install (from the repo root, board on USB):
#     mpremote cp test/gemm_protocol.py :
#     mpremote cp host/tt_um_sahpar_gemm4x4.py :
# Run on the board's REPL:
#     >>> import tt_um_sahpar_gemm4x4 as g
#     >>> g.main()                 # self-test on silicon
#     >>> g.demo_8x8()             # 8x8 int8 matmul on the 4x4 chip, checked
#
# Pins: the RP2 drives ui_in[7:0], uio[0] RX_VALID and uio[2] TX_READY; it
# reads uo_out[7:0], uio[1] RX_READY, uio[3] TX_VALID, uio[4] BUSY, uio[5] DONE.

import random

import microcotb as cocotb
from microcotb.clock import Clock
from microcotb.triggers import ClockCycles

cocotb.set_runner_scope(__name__)

from ttboard.demoboard import DemoBoard  # noqa: E402
from ttboard.mode import RPMode  # noqa: E402

import gemm_protocol as gp  # noqa: E402

PROJECT = "tt_um_sahpar_gemm4x4"
CLOCK_PERIOD_US = 10  # bit-banged project clock; the protocol is clock-agnostic


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_US, units="us").start())
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


def _rand_int8(count):
    return [random.randint(-128, 127) for _ in range(count)]


@cocotb.test()
async def test_id(dut):
    await reset(dut)
    got = await gp.read_id(dut)
    assert got == gp.CHIP_ID, "ID {:#x} != {:#x}".format(got, gp.CHIP_ID)


@cocotb.test()
async def test_status_idle(dut):
    await reset(dut)
    busy, done = await gp.status(dut)
    assert not busy and not done, "expected idle after reset"


@cocotb.test()
async def test_identity(dut):
    await reset(dut)
    ident = [1 if i == j else 0 for i in range(4) for j in range(4)]
    b = _rand_int8(16)
    await gp.gemm4(dut, ident, b)
    assert await gp.read_c(dut) == b


@cocotb.test()
async def test_random_4x4(dut):
    await reset(dut)
    random.seed(1)
    for _ in range(3):
        a, b = _rand_int8(16), _rand_int8(16)
        await gp.gemm4(dut, a, b)
        assert await gp.read_c(dut) == gp.matmul_ref(a, b, 4, 4, 4)


@cocotb.test()
async def test_accumulate_and_done(dut):
    await reset(dut)
    random.seed(2)
    a1, b1, a2, b2 = _rand_int8(16), _rand_int8(16), _rand_int8(16), _rand_int8(16)
    await gp.gemm4(dut, a1, b1)
    await gp.gemm4(dut, a2, b2, accumulate=True)
    exp = [x + y for x, y in zip(gp.matmul_ref(a1, b1, 4, 4, 4), gp.matmul_ref(a2, b2, 4, 4, 4))]
    assert await gp.read_c(dut) == exp
    assert int(dut.uio_out.value) & (1 << gp.DONE), "DONE pin should be high"


@cocotb.test()
async def test_tiled_8x8(dut):
    await reset(dut)
    random.seed(3)
    a, b = _rand_int8(64), _rand_int8(64)
    assert await gp.matmul(dut, a, b, 8, 8, 8) == gp.matmul_ref(a, b, 8, 8, 8)


@cocotb.test()
async def test_real_4x4(dut):
    """BF16 in, FP32 out: bit-exact against the host model of the chip."""
    await reset(dut)
    random.seed(4)
    a = [random.uniform(-3, 3) for _ in range(16)]
    b = [random.uniform(-3, 3) for _ in range(16)]
    assert await gp.gemm4_real(dut, a, b) == gp.gemm4_real_ref(a, b)


# ---------------------------------------------------------------------------
# demo: registered once, in its own runner scope so main() doesn't run it
DEMO_SCOPE = PROJECT + "_demo"
_demo = {}

cocotb.set_runner_scope(DEMO_SCOPE)


@cocotb.test()
async def demo_matmul(dut):
    await reset(dut)
    _demo["c"] = await gp.matmul(dut, _demo["a"], _demo["b"], 8, 8, 8)


cocotb.set_runner_scope(__name__)


# ---------------------------------------------------------------------------
def _setup():
    tt = DemoBoard.get()
    getattr(tt.shuttle, PROJECT).enable()
    if tt.mode != RPMode.ASIC_RP_CONTROL:
        tt.mode = RPMode.ASIC_RP_CONTROL
    tt.uio_oe_pico.value = gp.HOST_UIO_OE  # RX_VALID, TX_READY driven by us
    return tt


def _make_dut():
    import ttboard.cocotb.dut

    class DUT(ttboard.cocotb.dut.DUT):
        def __init__(self):
            super().__init__("gemm4x4")

    return DUT()


def main():
    """Run the self-tests above against the chip."""
    _setup()
    runner = cocotb.get_runner(__name__)
    runner.test(_make_dut())
    return runner


def demo_8x8(seed=42):
    """One 8x8 int8 matmul on silicon, printed and checked against Python."""
    _setup()
    random.seed(seed)
    _demo.clear()
    _demo["a"], _demo["b"] = _rand_int8(64), _rand_int8(64)
    cocotb.get_runner(DEMO_SCOPE).test(_make_dut())
    c = _demo.get("c")
    if c is None:
        print("demo did not complete")
        return None
    for r in range(8):
        print(" ".join("{:7d}".format(v) for v in c[r * 8:(r + 1) * 8]))
    print("matches reference:", c == gp.matmul_ref(_demo["a"], _demo["b"], 8, 8, 8))
    return c
