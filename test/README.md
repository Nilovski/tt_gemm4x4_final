# Testbench

cocotb tests for `tt_um_sahpar_gemm4x4`. `tb.v` wraps the design, `test.py`
holds the tests, and `gemm_protocol.py` is the host protocol and tiling
library shared with the demo-board runner in `../host`. Every result is
checked against a Python reference multiply.

## Running

RTL simulation:

```sh
make -B
```

Gate-level simulation: harden the project, copy the netlist from
`../runs/wokwi/results/final/verilog/gl/tt_um_sahpar_gemm4x4.v` to
`gate_level_netlist.v`, then run:

```sh
make -B GATES=yes
```

To write a VCD instead of FST, change `tb.v` to `$dumpfile("tb.vcd");` and
run `make -B FST=`.

## Viewing waveforms

```sh
gtkwave tb.fst tb.gtkw
# or
surfer tb.fst
```
