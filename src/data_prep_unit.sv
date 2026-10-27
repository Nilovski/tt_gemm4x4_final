`default_nettype none

// ---------------------------------------------------------------------------
// data_prep_unit -- operand buffers and skewed feed for the int8 array, with
// on-chip conversion of real (BF16) inputs.
//
// Buffers
// -------
// One K-deep lane per array edge input:
//
//   A lane r (west edge, row r):    A[r][0], A[r][1], ..., A[r][K-1]
//   B lane c (north edge, col c):   B[0][c], B[1][c], ..., B[K-1][c]
//
// Each slot is 16 bits, so a matrix can be loaded in either format:
//
//   int8 ("buffer data")  1 byte per element, used as-is.
//   BF16 ("real data")    2 bytes per element (low byte first), converted
//                         to int8 at the lane head as it is fed.
//
// Elements arrive in row-major order (A: e = r*K + k, B: e = k*N + c).
//
// Real data: block floating point
// -------------------------------
// While a BF16 matrix loads, the unit tracks the largest exponent EMAX of its
// non-zero elements. On feed, element x = (-1)^s * 1.m * 2^(e-127) becomes
//
//   q = (-1)^s * ({1, m} >> 1) >> (EMAX - e)        (|q| <= 127, truncated)
//
// so x ~= q * 2^(EMAX - 133). The matrix scale is EMAX - 133 (0 for int8),
// and C = A x B is recovered as C_int * 2^(scale_A + scale_B). The largest
// element keeps 7 significant bits; elements 2^7 or more times smaller than
// the largest flush to 0. Zero, subnormal (e = 0) inputs read as 0.
// Inf/NaN (e = 255) are not supported.
//
// Feed
// ----
// A one-cycle start pulse begins a feed of K + N - 1 cycles. On feed cycle t
// lane i presents its head while i <= t < i + K, so PE(i,j) sees A[i][k] and
// B[k][j] together on cycle k + i + j. Active lanes rotate by one each cycle
// and are back in their original order after the feed, so the same operands
// can be run again without reloading.
//
// Writes are ignored while feeding.
// ---------------------------------------------------------------------------
module data_prep_unit #(
    parameter int N  = 4,
    parameter int K  = 4,
    parameter int AW = $clog2(2 * N * K)    // byte address width
) (
    input  logic           clk,
    input  logic           rst_n,

    // loads from the command decoder
    input  logic           a_load_start,    // LOAD_A* opcode received
    input  logic           b_load_start,
    input  logic           load_real,       // format of the load being started
    input  logic           a_wr_en,
    input  logic           b_wr_en,
    input  logic [AW-1:0]  wr_addr,         // byte index within the matrix
    input  logic [7:0]     wr_data,

    // feed control
    input  logic           start,           // one-cycle pulse
    output logic           feeding,
    output logic           feed_done,

    // skewed int8 operands to the array edges
    output logic [N*8-1:0] a_west,
    output logic [N*8-1:0] b_north,

    // matrix scales (power of two), for C = C_int * 2^(scale_a + scale_b)
    output logic signed [9:0] scale_a,
    output logic signed [9:0] scale_b
);

    localparam int FEED_CYCLES = K + N - 1;
    localparam int TW          = (FEED_CYCLES > 1) ? $clog2(FEED_CYCLES) : 1;

    logic [15:0] a_lane [N][K];   // a_lane[r][k] = A[r][k] at rest
    logic [15:0] b_lane [N][K];   // b_lane[c][k] = B[k][c] at rest

    logic       a_real, b_real;   // format of the stored matrix
    logic [7:0] a_emax, b_emax;   // largest exponent seen (real format)
    logic       lo_b7;            // bit 7 of the previous (low) byte

    // ---- feed counter --------------------------------------------------
    logic          running;
    logic [TW-1:0] t_reg, t;

    assign feeding   = start | running;
    assign t         = running ? t_reg : '0;
    assign feed_done = feeding && (t == TW'(FEED_CYCLES - 1));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            running <= 1'b0;
            t_reg   <= '0;
        end else if (feeding) begin
            running <= !feed_done;
            t_reg   <= feed_done ? '0 : t + TW'(1);
        end
    end

    logic [N-1:0] active;
    always_comb
        for (int i = 0; i < N; i++)
            active[i] = feeding && (t >= TW'(i)) && (t < TW'(i + K));

    // ---- BF16 -> int8 at the lane heads --------------------------------
    function automatic logic [7:0] to_int8(input logic [15:0] x,
                                           input logic        is_real,
                                           input logic [7:0]  emax);
        logic [7:0] d;
        logic [6:0] mag;
        d   = emax - x[14:7];
        mag = (x[14:7] == 8'd0 || d > 8'd6) ? 7'd0
                                             : (7'({1'b1, x[6:1]}) >> d[2:0]);
        if (!is_real)    to_int8 = x[7:0];
        else if (x[15])  to_int8 = -{1'b0, mag};
        else             to_int8 = {1'b0, mag};
    endfunction

    always_comb begin
        for (int i = 0; i < N; i++) begin
            a_west[i*8 +: 8]  = active[i] ? to_int8(a_lane[i][0], a_real, a_emax) : 8'd0;
            b_north[i*8 +: 8] = active[i] ? to_int8(b_lane[i][0], b_real, b_emax) : 8'd0;
        end
    end

    assign scale_a = a_real ? $signed({2'b00, a_emax}) - 10'sd133 : 10'sd0;
    assign scale_b = b_real ? $signed({2'b00, b_emax}) - 10'sd133 : 10'sd0;

    // ---- byte address -> element and half ------------------------------
    // int8: element = addr, low half only. BF16: element = addr >> 1.
    logic [AW-1:0] a_e, b_e;
    logic          a_hi, b_hi;

    assign a_e  = a_real ? AW'(wr_addr[AW-1:1]) : wr_addr;
    assign a_hi = a_real & wr_addr[0];
    assign b_e  = b_real ? AW'(wr_addr[AW-1:1]) : wr_addr;
    assign b_hi = b_real & wr_addr[0];

    // exponent of the element completed by this (high) byte
    logic [7:0] new_e;
    assign new_e = {wr_data[6:0], lo_b7};

    // ---- load / rotate -------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_real <= 1'b0;
            b_real <= 1'b0;
            a_emax <= '0;
            b_emax <= '0;
            lo_b7  <= 1'b0;
            for (int i = 0; i < N; i++)
                for (int k = 0; k < K; k++) begin
                    a_lane[i][k] <= '0;
                    b_lane[i][k] <= '0;
                end
        end else if (feeding) begin
            for (int i = 0; i < N; i++)
                if (active[i]) begin
                    for (int k = 0; k < K - 1; k++) begin
                        a_lane[i][k] <= a_lane[i][k+1];
                        b_lane[i][k] <= b_lane[i][k+1];
                    end
                    a_lane[i][K-1] <= a_lane[i][0];
                    b_lane[i][K-1] <= b_lane[i][0];
                end
        end else begin
            if (a_load_start) begin
                a_real <= load_real;
                a_emax <= '0;
            end
            if (b_load_start) begin
                b_real <= load_real;
                b_emax <= '0;
            end

            if (a_wr_en || b_wr_en) lo_b7 <= wr_data[7];

            if (a_wr_en && a_real && a_hi && new_e > a_emax) a_emax <= new_e;
            if (b_wr_en && b_real && b_hi && new_e > b_emax) b_emax <= new_e;

            for (int i = 0; i < N; i++)
                for (int k = 0; k < K; k++) begin
                    // A element r*K + k -> a_lane[r][k]
                    if (a_wr_en && a_e == AW'(i*K + k)) begin
                        if (a_hi)        a_lane[i][k][15:8] <= wr_data;
                        else begin
                            a_lane[i][k][7:0] <= wr_data;
                            if (!a_real) a_lane[i][k][15:8] <= '0;
                        end
                    end
                    // B element k*N + c -> b_lane[c][k]
                    if (b_wr_en && b_e == AW'(k*N + i)) begin
                        if (b_hi)        b_lane[i][k][15:8] <= wr_data;
                        else begin
                            b_lane[i][k][7:0] <= wr_data;
                            if (!b_real) b_lane[i][k][15:8] <= '0;
                        end
                    end
                end
        end
    end

endmodule

`default_nettype wire
