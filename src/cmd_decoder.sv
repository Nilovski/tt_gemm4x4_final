`default_nettype none

// ---------------------------------------------------------------------------
// cmd_decoder -- byte-protocol front end for the N x N GEMM core. Opcodes:
//
//   0x00  NOP
//   0x10  LOAD_A          -- followed by N*K int8 bytes (row-major)
//   0x11  LOAD_B          -- followed by K*N int8 bytes (row-major)
//   0x12  LOAD_A_REAL     -- followed by N*K BF16 values, 2 bytes each,
//                            low byte first (row-major)
//   0x13  LOAD_B_REAL     -- followed by K*N BF16 values, as above
//   0x20  GEMM            -- pulses start (ignored while busy); C += A x B
//   0x21  CLEAR_GEMM      -- pulses start (ignored while busy); C  = A x B
//   0x30  READ_C          -- streams all of C: N*N elements, row-major,
//                            4 bytes each (signed int32, low byte first)
//   0x31  READ_C_ELEMENT  -- followed by 1 address byte (elem*4 + byte),
//                            replies with that byte of C as signed int32
//   0x32  READ_C_REAL     -- followed by 1 address byte (elem*4 + byte),
//                            replies with that byte of C as an FP32 value,
//                            C_int * 2^(scale_A + scale_B)
//   0x33  READ_C_REAL_ALL -- streams all of C as FP32, same order as READ_C
//   0x40  STATUS          -- replies
//                            bit 0 busy
//                            bit 1 done: sticky, set when a run finishes,
//                                  cleared by the next GEMM/CLEAR_GEMM (pin)
//                            bit 2 A_loaded: a complete A has been loaded
//                            bit 3 B_loaded: a complete B has been loaded
//                            bit 4 result_valid: C holds a finished result
//                            bit 5 error: sticky, cleared by CLEAR_GEMM. Set
//                                  when a LOAD/GEMM/CLEAR_GEMM arrives while
//                                  busy (ignored), or when a GEMM accumulates
//                                  with a different A/B scale than the run it
//                                  adds to (READ_C_REAL then not meaningful)
//                            bits 7:6 reserved (0)
//   0x41  ID              -- replies CHIP_ID
//
// Anything else is ignored.
//
// Implementation notes:
//   * Operands live in data_prep_unit; this module only steers bytes to it.
//     Data bytes of a LOAD that arrives while busy are consumed (so the byte
//     stream stays in sync) but discarded.
//   * C is read straight out of the PE accumulators, which hold their value
//     after done because the array is only enabled while busy. Reading
//     during a run returns a partial sum, so poll STATUS first.
//   * Accumulators are ACC_W bits; READ_C_ELEMENT sign-extends each element
//     to 32 bits. With GEMM (accumulate) the host must keep the total number
//     of summed products within ACC_W (see top level).
// ---------------------------------------------------------------------------
module cmd_decoder #(
    parameter int N        = 4,
    parameter int K        = 4,
    parameter int ACC_W    = 21,
    parameter logic [7:0] CHIP_ID = 8'hA8,
    parameter int AW       = $clog2(2 * N * K)
) (
    input  logic clk,
    input  logic rst_n,

    // from rx_byte_interface
    input  logic [7:0] rx_byte,
    input  logic       rx_byte_valid,

    // to tx_byte_interface
    output logic [7:0] tx_byte,
    output logic       tx_byte_valid,
    input  logic       tx_ready,

    // data_prep_unit loads
    output logic          a_load_start,
    output logic          b_load_start,
    output logic          load_real,
    output logic          a_wr_en,
    output logic          b_wr_en,
    output logic [AW-1:0] wr_addr,
    output logic [7:0]    wr_data,
    input  logic signed [9:0] scale_a,
    input  logic signed [9:0] scale_b,

    // gemm_top
    output logic                  gemm_start,
    output logic                  gemm_accumulate,
    input  logic                  gemm_busy,
    input  logic                  gemm_done,
    input  logic [N*N*ACC_W-1:0]  c_flat,

    output logic                  result_done   // sticky DONE flag
);

    localparam int NUM_INT_BYTES  = N*K;        // one matrix, int8
    localparam int NUM_REAL_BYTES = 2*N*K;      // one matrix, BF16
    localparam int NUM_C_ELEMS    = N*N;
    localparam int EW = (NUM_C_ELEMS > 1) ? $clog2(NUM_C_ELEMS) : 1;

    localparam logic [7:0] OP_NOP            = 8'h00;
    localparam logic [7:0] OP_LOAD_A         = 8'h10;
    localparam logic [7:0] OP_LOAD_B         = 8'h11;
    localparam logic [7:0] OP_LOAD_A_REAL    = 8'h12;
    localparam logic [7:0] OP_LOAD_B_REAL    = 8'h13;
    localparam logic [7:0] OP_GEMM           = 8'h20;
    localparam logic [7:0] OP_CLEAR_GEMM     = 8'h21;
    localparam logic [7:0] OP_READ_C         = 8'h30;
    localparam logic [7:0] OP_READ_C_ELEMENT = 8'h31;
    localparam logic [7:0] OP_READ_C_REAL    = 8'h32;
    localparam logic [7:0] OP_READ_C_REAL_ALL = 8'h33;
    localparam logic [7:0] OP_STATUS         = 8'h40;
    localparam logic [7:0] OP_ID             = 8'h41;

    typedef enum logic [1:0] {
        S_IDLE,
        S_LOAD,
        S_READ_ADDR,
        S_READ_ALL
    } state_t;

    state_t        state;
    logic          load_b;      // current load targets B
    logic          load_drop;   // LOAD started while busy: swallow its data
    logic [AW-1:0] idx;
    logic [AW-1:0] last_idx;
    logic          read_real;

    logic signed [10:0] c_scale;       // scale of the values in C
    logic               scale_mismatch;
    logic               cmd_dropped;    // command ignored because busy
    logic               a_loaded, b_loaded;
    localparam int RW = $clog2(4 * NUM_C_ELEMS);
    logic [RW-1:0]      rd_cnt;         // READ_C stream position
    logic signed [10:0] run_scale;

    assign run_scale = 11'(scale_a) + 11'(scale_b);

    // ---- byte steering to the prep unit --------------------------------
    assign wr_addr = idx;
    assign wr_data = rx_byte;
    assign a_wr_en = (state == S_LOAD) && rx_byte_valid && !load_drop && !load_b;
    assign b_wr_en = (state == S_LOAD) && rx_byte_valid && !load_drop &&  load_b;

    logic is_load;
    assign is_load = rx_byte == OP_LOAD_A || rx_byte == OP_LOAD_B ||
                     rx_byte == OP_LOAD_A_REAL || rx_byte == OP_LOAD_B_REAL;

    assign a_load_start = (state == S_IDLE) && rx_byte_valid && !gemm_busy &&
                          (rx_byte == OP_LOAD_A || rx_byte == OP_LOAD_A_REAL);
    assign b_load_start = (state == S_IDLE) && rx_byte_valid && !gemm_busy &&
                          (rx_byte == OP_LOAD_B || rx_byte == OP_LOAD_B_REAL);
    assign load_real    = rx_byte[1];

    // ---- C readout: element select, 32-bit word, byte select ------------
    logic [EW-1:0]           rd_elem;
    logic [1:0]              rd_byte;
    logic signed [ACC_W-1:0] rd_acc;
    logic [31:0]             rd_int, rd_fp, rd_word;
    logic [7:0]              rd_data;

    logic [7:0] rd_addr;
    assign rd_addr = (state == S_READ_ALL) ? 8'(rd_cnt) : rx_byte;
    assign rd_elem = rd_addr[EW+1:2];
    assign rd_byte = rd_addr[1:0];
    assign rd_acc  = c_flat[rd_elem*ACC_W +: ACC_W];
    assign rd_int  = 32'(rd_acc);                 // sign-extending cast

    int_to_fp32 #(.XW(ACC_W)) u_to_fp (
        .x     (rd_acc),
        .scale (c_scale[9:0]),
        .y     (rd_fp)
    );

    assign rd_word = read_real ? rd_fp : rd_int;
    assign rd_data = rd_word[rd_byte*8 +: 8];

    // Single-byte replies go out directly (the host drains each one before
    // sending the next command); READ_C streams wait for tx_ready per byte.

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            load_b          <= 1'b0;
            load_drop       <= 1'b0;
            idx             <= '0;
            last_idx        <= '0;
            read_real       <= 1'b0;
            gemm_start      <= 1'b0;
            gemm_accumulate <= 1'b0;
            result_done     <= 1'b0;
            c_scale         <= '0;
            scale_mismatch  <= 1'b0;
            cmd_dropped     <= 1'b0;
            a_loaded        <= 1'b0;
            b_loaded        <= 1'b0;
            rd_cnt          <= '0;
            tx_byte         <= '0;
            tx_byte_valid   <= 1'b0;
        end else begin
            gemm_start    <= 1'b0;
            tx_byte_valid <= 1'b0;

            // sticky DONE: set on completion, cleared when a new run is issued
            if (gemm_done)       result_done <= 1'b1;
            else if (gemm_start) result_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (rx_byte_valid) begin
                        if (is_load) begin
                            state     <= S_LOAD;
                            load_b    <= rx_byte[0];
                            load_drop <= gemm_busy;
                            if (gemm_busy) cmd_dropped <= 1'b1;
                            idx       <= '0;
                            last_idx  <= rx_byte[1] ? AW'(NUM_REAL_BYTES - 1)
                                                    : AW'(NUM_INT_BYTES - 1);
                        end
                        case (rx_byte)
                            OP_NOP: ;
                            OP_ID: begin
                                tx_byte       <= CHIP_ID;
                                tx_byte_valid <= 1'b1;
                            end
                            OP_STATUS: begin
                                tx_byte       <= {2'b0, scale_mismatch | cmd_dropped,
                                                  result_done, b_loaded, a_loaded,
                                                  result_done, gemm_busy};
                                tx_byte_valid <= 1'b1;
                            end
                            OP_GEMM, OP_CLEAR_GEMM: begin
                                if (!gemm_busy) begin
                                    gemm_start      <= 1'b1;
                                    gemm_accumulate <= (rx_byte == OP_GEMM);
                                    c_scale         <= run_scale;
                                    if (rx_byte == OP_CLEAR_GEMM) begin
                                        scale_mismatch <= 1'b0;
                                        cmd_dropped    <= 1'b0;
                                    end else if (run_scale != c_scale)
                                        scale_mismatch <= 1'b1;
                                end else begin
                                    cmd_dropped <= 1'b1;
                                end
                            end
                            OP_READ_C, OP_READ_C_REAL_ALL: begin
                                state     <= S_READ_ALL;
                                read_real <= rx_byte[0];
                                rd_cnt    <= '0;
                            end
                            OP_READ_C_ELEMENT, OP_READ_C_REAL: begin
                                state     <= S_READ_ADDR;
                                read_real <= rx_byte[1];
                            end
                            default: ;
                        endcase
                    end
                end

                S_LOAD: begin
                    if (rx_byte_valid) begin
                        if (idx == last_idx) begin
                            state <= S_IDLE;
                            if (!load_drop) begin
                                if (load_b) b_loaded <= 1'b1;
                                else        a_loaded <= 1'b1;
                            end
                        end else begin
                            idx <= idx + AW'(1);
                        end
                    end
                end

                S_READ_ADDR: begin
                    if (rx_byte_valid) begin
                        tx_byte       <= rd_data;
                        tx_byte_valid <= 1'b1;
                        state         <= S_IDLE;
                    end
                end

                // Stream one byte whenever the TX holding register is free.
                // Bytes from the host during the stream are ignored.
                S_READ_ALL: begin
                    if (tx_ready && !tx_byte_valid) begin
                        tx_byte       <= rd_data;
                        tx_byte_valid <= 1'b1;
                        if (rd_cnt == RW'(4 * NUM_C_ELEMS - 1)) state  <= S_IDLE;
                        else                                    rd_cnt <= rd_cnt + RW'(1);
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // address bits above the valid range (addr >= NUM_C_BYTES) alias/wrap
    wire _unused = &{rd_addr[7:EW+2], 1'b0};

endmodule

`default_nettype wire
