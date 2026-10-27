`default_nettype none

// ---------------------------------------------------------------------------
// tx_byte_interface -- chip -> host byte sender.
//
// The driver hands off one byte at a time with a single tx_byte_valid pulse
// (only ever asserted when tx_ready_o says the register is free). The byte
// is held on host_data with host_valid high until the host asserts
// host_ready (TX_READY), at which point it is dropped.
// ---------------------------------------------------------------------------
module tx_byte_interface (
    input  logic       clk,
    input  logic       rst_n,

    input  logic [7:0] tx_byte,
    input  logic       tx_byte_valid,
    output logic       tx_ready_o,     // 1 when a new byte can be accepted

    input  logic       host_ready,     // TX_READY, from host

    output logic [7:0] host_data,      // -> uo_out
    output logic       host_valid      // -> TX_VALID
);

    logic [7:0] byte_reg;
    logic       pending;

    assign tx_ready_o = !pending;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            byte_reg <= '0;
            pending  <= 1'b0;
        end else begin
            if (tx_byte_valid && !pending) begin
                byte_reg <= tx_byte;
                pending  <= 1'b1;
            end else if (pending && host_ready) begin
                pending  <= 1'b0;
            end
        end
    end

    assign host_data  = byte_reg;
    assign host_valid = pending;

endmodule

`default_nettype wire
