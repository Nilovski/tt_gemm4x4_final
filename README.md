![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg)

# 4x4 integer systolic GEMM

A 4x4 output-stationary signed int8 systolic array for Tiny Tapeout (sky130),
with a clear mode (`C = A x B`) and an accumulate mode (`C += A x B`), driven
over an 8-bit byte-serial host protocol. The host tiles matrix products of any
size onto the array, e.g. an 8x8 product as 4 output blocks x 2 K-blocks.

- Top module: `tt_um_sahpar_gemm4x4`
- Tiles: 4x2
- Chip ID: `0xA4`
- Protocol, pinout and test procedure: [docs/info.md](docs/info.md)

## Features

- 16 processing elements with 8x8 signed multipliers and 21-bit accumulators
  (exact for any sum of up to 63 products), sign-extended to 32 bits on readout
- Counter-scheduled operand skew; results are read directly from the accumulators
- Row-major A and B, no host-side transpose
- `GEMM` (0x20) accumulates onto the previous result for tiled matmuls;
  `CLEAR_GEMM` (0x21) starts a new one
- `BUSY` (`uio[4]`) and sticky `DONE` (`uio[5]`) outputs; DONE can be used as
  an interrupt line
- LOAD commands received while busy are consumed and discarded, so stored
  operands stay intact and the byte stream stays in sync

## Repository layout

```
src/    RTL: tt_um_sahpar_gemm4x4 -> cmd_decoder + rx/tx_byte_interface -> gemm_top -> systolic_array -> pe -> mac_unit
test/   cocotb testbench (test.py) and gemm_protocol.py, the host protocol and tiling library
host/   tt_um_sahpar_gemm4x4.py, demo-board runner (MicroPython + microcotb) built on gemm_protocol.py
docs/   datasheet (info.md)
```

## Running the tests

```sh
pip install -r test/requirements.txt
cd test
make -B
```

The suite covers the ID and pin directions, idle STATUS, identity and random
products, accumulator extremes, operand reuse, LOAD/CLEAR while busy, `ena`
low, DONE behaviour, accumulate vs. clear, the 21-bit accumulation limit, and
host-tiled 8x8, 5x7 @ 7x3 and inner-dimension-64 products. Gate-level
simulation runs in CI after hardening (`make -B GATES=yes`).

## Running on the demo board

Copy `test/gemm_protocol.py` and `host/tt_um_sahpar_gemm4x4.py` to the RP2
with `mpremote`, then in the REPL:

```python
import tt_um_sahpar_gemm4x4 as g
g.main()        # self-tests: ID, identity, random, accumulate, tiled 8x8
g.demo_8x8()    # prints an 8x8 int8 matmul computed on the 4x4 array
```

## Build

Every push runs the GitHub Actions workflows: `test`, then `gds` followed by
`precheck`, `gl_test` and `viewer`. For `viewer`, set Settings → Pages →
Source to **GitHub Actions**.

## License

Apache 2.0. See [LICENSE](LICENSE).
