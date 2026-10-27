## How it works

A 4x4 output-stationary integer systolic array computes **C = A x B** or
**C += A x B**, where A (4x4) and B (4x4) hold signed 8-bit values. A flows in
from the west edge and B from the north edge. Row *i* and column *j* are
delayed by *i* and *j* cycles, so each processing element PE(i,j) receives
A[i][k] and B[k][j] on the same cycle and accumulates C[i][j] in place. A run
takes K + 2(N-1) = 10 clock cycles.

A **data preparation unit** sits in front of the array. It holds A and B in
4 + 4 lane buffers (one per array row and column, 4 values deep) and, on the
run's start signal, feeds them to the array edges with the skew above. The
lanes rotate back to their original order after each run, so the same
operands can be run again without reloading.

Operands can be loaded in two formats:

- **int8** ("buffer data", `LOAD_A`/`LOAD_B`): 1 byte per value, used as-is.
- **Real numbers** (`LOAD_A_REAL`/`LOAD_B_REAL`): BF16, 2 bytes per value.
  The data preparation unit converts each matrix to int8 with a shared
  power-of-two scale (block floating point): it tracks the largest exponent
  while the matrix loads, then turns every value into a 7-bit magnitude plus
  sign relative to it. `READ_C_REAL` returns C as FP32, with both scales
  applied. Each value keeps up to 7 significant bits (truncated), and values
  128 or more times smaller than the matrix's largest flush to zero. Inf and
  NaN are not supported.

Because results can accumulate across runs, a host can compute a matrix
product of **any size** on this 4x4 array. It splits the operands into 4x4
blocks and sums the K-blocks for each output block on-chip.

Two synchronous handshake channels carry bytes. `RX_VALID` + `ui[7:0]` bring
a byte in (`RX_READY` is always high), and `TX_VALID` + `uo[7:0]` send one
out, held until the host pulses `TX_READY`.

| Byte   | Op             | Followed by                         | Reply                   |
|--------|----------------|-------------------------------------|-------------------------|
| `0x00` | NOP            | --                                  | none                    |
| `0x10` | LOAD_A         | 16 bytes, row-major a00..a33        | none                    |
| `0x11` | LOAD_B         | 16 bytes, row-major b00..b33        | none                    |
| `0x12` | LOAD_A_REAL    | 32 bytes: 16 BF16, low byte first   | none                    |
| `0x13` | LOAD_B_REAL    | 32 bytes: 16 BF16, low byte first   | none                    |
| `0x20` | GEMM           | --                                  | none (C += A x B)       |
| `0x21` | CLEAR_GEMM     | --                                  | none (C = A x B)        |
| `0x31` | READ_C_ELEMENT | 1 address byte, 0-63                | 1 byte of C (int32)     |
| `0x32` | READ_C_REAL    | 1 address byte, 0-63                | 1 byte of C (FP32)      |
| `0x40` | STATUS         | --                                  | `{5'b0, scale_mismatch, done, busy}` |
| `0x41` | ID             | --                                  | `0xA4`                  |

- C is read as 16 elements x 4 bytes: signed 32-bit, little-endian. The
  address is `elem*4 + byte`, and `elem` goes in row-major order (c00, c01,
  ... c33).
- The accumulators are 21 bits, sign-extended to 32 on readout. They hold any
  sum of up to **63 products** exactly. That means one CLEAR_GEMM plus up to
  14 GEMMs (inner dimension up to 60) before C must be read out. For longer
  inner dimensions the host adds the partial results itself.
- A and B are held between runs, so you can reload just one of them. Each
  keeps its own format, so int8 and real operands can be mixed (an int8
  matrix has scale 1).
- With real data, GEMM (accumulate) is only meaningful when the new A and B
  have the same combined scale as the run being added to; otherwise STATUS
  bit 2 (scale_mismatch) is set until the next CLEAR_GEMM. The host helper
  `matmul_real` therefore runs each K-block with CLEAR_GEMM and sums the
  FP32 results itself.
- GEMM and CLEAR_GEMM are ignored while busy. A LOAD sent while busy has its
  data bytes consumed and discarded, so the stored operands stay intact and
  the byte stream stays in sync.
- **BUSY** (`uio[4]`) is high while a run is in progress.
- **DONE** (`uio[5]`) goes high when a run finishes and stays high until the
  next GEMM or CLEAR_GEMM. It works as an interrupt line. STATUS reports both
  flags.

## How to test

1. Reset, then send `0x41` (ID) and expect `0xA4`.
2. Send `0x10` and then 16 bytes of A. Send `0x11` and then 16 bytes of B.
   Negative values are two's complement.
3. Send `0x21` (CLEAR_GEMM) to start.
4. Wait for `uio[5]` DONE, or poll `0x40` until bit 0 (busy) reads 0.
5. For each address 0-63, send `0x31` and then the address, and collect the
   reply byte. Reassemble the 16 little-endian 32-bit values.

To accumulate, load new A/B blocks and send `0x20` instead of `0x21`.

For real numbers, use `0x12`/`0x13` with 16 BF16 values (2 bytes each, low
byte first) instead of step 2, and `0x32` instead of `0x31` in step 5; the
16 values read back are IEEE-754 singles.

**On the Tiny Tapeout demo board**, copy `test/gemm_protocol.py` and
`host/tt_um_sahpar_gemm4x4.py` to the RP2 with `mpremote`. Then, in the REPL:

```python
import tt_um_sahpar_gemm4x4 as g
g.main()        # self-tests on silicon (ID, identity, random, accumulate, tiled 8x8)
g.demo_8x8()    # prints an 8x8 int8 matmul computed on the 4x4 chip
```

These scripts use the same protocol and tiling code as the cocotb simulation
tests in `test/test.py`.

## External hardware

None. All you need is a host (the demo board's RP2040/RP2350, or any MCU or
FPGA) driving `ui`/`uo`/`uio[5:0]`.
