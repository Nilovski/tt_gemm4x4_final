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
//   0x31  READ_C_ELEMENT  -- followed by 1 address byte (elem*4 + byte),
//                            replies with that byte of C as signed int32
//   0x32  READ_C_REAL     -- followed by 1 address byte (elem*4 + byte),
//                            replies with that byte of C as an FP32 value,
//                            C_int * 2^(scale_A + scale_B)
//   0x40  STATUS          -- replies {5'b0, scale_mismatch, done, busy}
//                            done: sticky, set when a run finishes, cleared
//                            by the next GEMM/CLEAR_GEMM (also on a pin)
//                            scale_mismatch: a GEMM (accumulate) ran with a
//                            different A/B scale than the run it adds to, so
//                            READ_C_REAL is not meaningful; cleared by
//                            CLEAR_GEMM
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
    parameter logic [7:0] CHIP_ID = 8'hA4,
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
    localparam logic [7:0] OP_READ_C_ELEMENT = 8'h31;
    localparam logic [7:0] OP_READ_C_REAL    = 8'h32;
    localparam logic [7:0] OP_STATUS         = 8'h40;
    localparam logic [7:0] OP_ID             = 8'h41;

    typedef enum logic [1:0] {
        S_IDLE,
        S_LOAD,
        S_READ_ADDR
    } state_t;

    state_t        state;
    logic          load_b;      // current load targets B
    logic          load_drop;   // LOAD started while busy: swallow its data
    logic [AW-1:0] idx;
    logic [AW-1:0] last_idx;
    logic          read_real;

    logic signed [10:0] c_scale;       // scale of the values in C
    logic               scale_mismatch;
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

    assign rd_elem = rx_byte[EW+1:2];
    assign rd_byte = rx_byte[1:0];
    assign rd_acc  = c_flat[rd_elem*ACC_W +: ACC_W];
    assign rd_int  = 32'(rd_acc);                 // sign-extending cast

    int_to_fp32 #(.XW(ACC_W)) u_to_fp (
        .x     (rd_acc),
        .scale (c_scale[9:0]),
        .y     (rd_fp)
    );

    assign rd_word = read_real ? rd_fp : rd_int;
    assign rd_data = rd_word[rd_byte*8 +: 8];

    // At most one reply is issued per command and the host drains it before
    // sending the next command, so tx_ready back-pressure is not modelled.

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
                                tx_byte       <= {5'b0, scale_mismatch, result_done, gemm_busy};
                                tx_byte_valid <= 1'b1;
                            end
                            OP_GEMM, OP_CLEAR_GEMM: begin
                                if (!gemm_busy) begin
                                    gemm_start      <= 1'b1;
                                    gemm_accumulate <= (rx_byte == OP_GEMM);
                                    c_scale         <= run_scale;
                                    if (rx_byte == OP_CLEAR_GEMM)
                                        scale_mismatch <= 1'b0;
                                    else if (run_scale != c_scale)
                                        scale_mismatch <= 1'b1;
                                end
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
                        if (idx == last_idx) state <= S_IDLE;
                        else                 idx   <= idx + AW'(1);
                    end
                end

                S_READ_ADDR: begin
                    if (rx_byte_valid) begin
                        tx_byte       <= rd_data;
                        tx_byte_valid <= 1'b1;
                        state         <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // address bits above the valid range (addr >= NUM_C_BYTES) alias/wrap
    wire _unused = &{tx_ready, rx_byte[7:EW+2], 1'b0};

endmodule

`default_nettype wire
