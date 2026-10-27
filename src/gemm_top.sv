`default_nettype none

// ---------------------------------------------------------------------------
// gemm_top -- run sequencing for C = A * B on an N x N output-stationary
// systolic array.
//
// The operands come in already skewed from data_prep_unit: on run cycle t,
// row i carries A[i][t-i] and column j carries B[t-j][j] (zero outside
// 0 <= t-i < K), so PE(i,j) sees A[i][k] and B[k][j] together on cycle
// k + i + j. first_cycle marks run cycle 0 and is the prep unit's start.
//
// Run length T = K + 2*(N-1): K cycles of real data plus the time for the
// last operand to reach PE(N-1,N-1). Trailing cycles carry zeros, which are
// a no-op for the accumulator.
//
// Handshake: pulse start (busy must be low), wait for done, read c_flat.
//
// accumulate (sampled with start): 0 -> C = A*B (accumulators cleared on the
// first cycle); 1 -> C += A*B (accumulators keep their previous value). The
// accumulate form lets a host compute larger matrices by tiling: sum several
// K-blocks into the same C block before reading it out.
// ---------------------------------------------------------------------------
module gemm_top #(
    parameter int N      = 4,
    parameter int K      = 4,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32,
    // Extra cycles the array is held enabled after the last real operand, so a
    // pipelined MAC can drain before done is asserted. 0 for the
    // combinational integer MAC.
    parameter int MAC_LATENCY = 0
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start,
    input  logic                    accumulate,
    input  logic [N*DATA_W-1:0]     a_west,
    input  logic [N*DATA_W-1:0]     b_north,
    output logic                    first_cycle,
    output logic                    busy,
    output logic                    done,
    output logic [N*N*ACC_W-1:0]    c_flat
);

    localparam int T  = K + 2*(N-1) + MAC_LATENCY;
    localparam int CW = (T > 1) ? $clog2(T) : 1;

    // ---- run counter ------------------------------------------------------
    logic [CW-1:0] t;
    logic          acc_run;   // current run accumulates onto the previous C

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy    <= 1'b0;
            done    <= 1'b0;
            t       <= '0;
            acc_run <= 1'b0;
        end else begin
            done <= 1'b0;
            if (!busy) begin
                if (start) begin
                    busy    <= 1'b1;
                    t       <= '0;
                    acc_run <= accumulate;
                end
            end else begin
                if (t == CW'(T-1)) begin
                    busy <= 1'b0;
                    done <= 1'b1;      // c_flat valid from this cycle onward
                end else begin
                    t <= t + CW'(1);
                end
            end
        end
    end

    assign first_cycle = busy && (t == '0);

    // ---- the array --------------------------------------------------------
    systolic_array #(
        .N      (N),
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W)
    ) u_array (
        .clk     (clk),
        .rst_n   (rst_n),
        // first cycle of a clearing run overwrites; an accumulating run never
        // clears (edge inputs are zero until real data arrives, so the extra
        // cycles add nothing)
        .clr     (first_cycle && !acc_run),
        .en      (busy),
        .a_west  (a_west),
        .b_north (b_north),
        .c_flat  (c_flat)
    );

endmodule

`default_nettype wire
