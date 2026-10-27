`default_nettype none

// ---------------------------------------------------------------------------
// tt_um_sahpar_gemm4x4 -- Tiny Tapeout top level for the 4x4 int8
// output-stationary systolic GEMM.
//
//   host bytes -> rx -> cmd_decoder -> data_prep_unit -> gemm_top (array)
//   host bytes <- tx <- cmd_decoder <- C readout (int32 or FP32)
//
// Operands load as int8 ("buffer data") or BF16 ("real data"); the data
// preparation unit converts BF16 to int8 with a per-matrix power-of-two
// scale, and READ_C_REAL returns results as FP32.
//
// Pin map:
//   ui_in[7:0]  = host_data in   (RX byte)
//   uo_out[7:0] = host_data out  (TX byte)
//   uio[0] = RX_VALID   in   (host asserts when ui_in holds a new byte)
//   uio[1] = RX_READY   out  (always 1 -- see rx_byte_interface)
//   uio[2] = TX_READY   in   (host asserts to accept the byte on uo_out)
//   uio[3] = TX_VALID   out  (chip asserts while uo_out holds an unread byte)
//   uio[4] = BUSY       out  (GEMM running; STATUS bit 0)
//   uio[5] = DONE       out  (sticky: a run finished and C is ready; cleared by
//                             the next GEMM/CLEAR_GEMM; STATUS bit 1). Usable
//                             as an interrupt line on the host.
//   uio[7:6] reserved (inputs)
// ---------------------------------------------------------------------------
module tt_um_sahpar_gemm4x4 (
    input  logic [7:0] ui_in,
    output logic [7:0] uo_out,
    input  logic [7:0] uio_in,
    output logic [7:0] uio_out,
    output logic [7:0] uio_oe,
    input  logic       ena,
    input  logic       clk,
    input  logic       rst_n
);

    localparam int N      = 4;
    localparam int K      = 4;
    localparam int DATA_W = 8;
    // Each signed 8x8 product lies in [-16256, 16384]. 21-bit accumulators
    // hold any sum of up to 63 products exactly (|sum| <= 1,032,192 < 2^20),
    // i.e. one CLEAR_GEMM plus up to 14 chained GEMM (accumulate) runs of K=4:
    // enough to tile matmuls with inner dimension up to 60 (8x8, 16x16,
    // 32x32, ...) on the host. Readout sign-extends to 32 bits.
    localparam int ACC_W  = 21;

    // ---- RX ----------------------------------------------------------
    logic       rx_byte_valid;
    logic       rx_host_ready;
    logic [7:0] rx_byte;

    rx_byte_interface u_rx (
        .clk           (clk),
        .rst_n         (rst_n),
        .host_data     (ui_in),
        .host_valid    (uio_in[0]),
        .host_ready    (rx_host_ready),
        .rx_byte       (rx_byte),
        .rx_byte_valid (rx_byte_valid)
    );

    // ---- TX ------------------------------------------------------------
    logic [7:0] tx_byte;
    logic       tx_byte_valid;
    logic       tx_ready;
    logic       tx_host_valid;

    tx_byte_interface u_tx (
        .clk           (clk),
        .rst_n         (rst_n),
        .tx_byte       (tx_byte),
        .tx_byte_valid (tx_byte_valid),
        .tx_ready_o    (tx_ready),
        .host_ready    (uio_in[2]),
        .host_data     (uo_out),
        .host_valid    (tx_host_valid)
    );

    // ---- decoder -------------------------------------------------------
    localparam int AW = $clog2(2 * N * K);

    logic                  gemm_start, gemm_accumulate, gemm_busy, gemm_done;
    logic                  result_done;
    logic                  a_load_start, b_load_start, load_real;
    logic                  a_wr_en, b_wr_en;
    logic [AW-1:0]         wr_addr;
    logic [7:0]            wr_data;
    logic signed [9:0]     scale_a, scale_b;
    logic [N*N*ACC_W-1:0]  c_flat;

    cmd_decoder #(
        .N(N), .K(K), .ACC_W(ACC_W), .CHIP_ID(8'hA4), .AW(AW)
    ) u_decoder (
        .clk             (clk),
        .rst_n           (rst_n),
        .rx_byte         (rx_byte),
        .rx_byte_valid   (rx_byte_valid & ena),
        .tx_byte         (tx_byte),
        .tx_byte_valid   (tx_byte_valid),
        .tx_ready        (tx_ready),
        .a_load_start    (a_load_start),
        .b_load_start    (b_load_start),
        .load_real       (load_real),
        .a_wr_en         (a_wr_en),
        .b_wr_en         (b_wr_en),
        .wr_addr         (wr_addr),
        .wr_data         (wr_data),
        .scale_a         (scale_a),
        .scale_b         (scale_b),
        .gemm_start      (gemm_start),
        .gemm_accumulate (gemm_accumulate),
        .gemm_busy       (gemm_busy),
        .gemm_done       (gemm_done),
        .c_flat          (c_flat),
        .result_done     (result_done)
    );

    // ---- data preparation unit -------------------------------------------
    // A/B lane buffers (int8 or BF16), BF16 -> int8 conversion and the
    // skewed feed. gemm_top's first run cycle is its start signal.
    logic                  feed_start, feeding, feed_done;
    logic [N*DATA_W-1:0]   a_west, b_north;

    data_prep_unit #(.N(N), .K(K), .AW(AW)) u_prep (
        .clk          (clk),
        .rst_n        (rst_n),
        .a_load_start (a_load_start),
        .b_load_start (b_load_start),
        .load_real    (load_real),
        .a_wr_en      (a_wr_en),
        .b_wr_en      (b_wr_en),
        .wr_addr      (wr_addr),
        .wr_data      (wr_data),
        .start        (feed_start),
        .feeding      (feeding),
        .feed_done    (feed_done),
        .a_west       (a_west),
        .b_north      (b_north),
        .scale_a      (scale_a),
        .scale_b      (scale_b)
    );

    // ---- compute core ----------------------------------------------------
    gemm_top #(
        .N(N), .K(K), .DATA_W(DATA_W), .ACC_W(ACC_W), .MAC_LATENCY(0)
    ) u_gemm (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (gemm_start),
        .accumulate  (gemm_accumulate),
        .a_west      (a_west),
        .b_north     (b_north),
        .first_cycle (feed_start),
        .busy        (gemm_busy),
        .done        (gemm_done),
        .c_flat      (c_flat)
    );

    assign uio_out = {2'b0, result_done, gemm_busy, tx_host_valid, 1'b0, rx_host_ready, 1'b0};
    assign uio_oe  = 8'b0011_1010;   // [5]=DONE, [4]=BUSY, [3]=TX_VALID, [1]=RX_READY

    wire _unused = &{uio_in[7:3], uio_in[1], feeding, feed_done, 1'b0};

endmodule

`default_nettype wire
