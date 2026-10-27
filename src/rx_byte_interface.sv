`default_nettype none

// ---------------------------------------------------------------------------
// rx_byte_interface -- host -> chip byte receiver.
//
// host_ready is tied to 1'b1: every cycle
// host_valid is asserted, the byte on host_data is captured and pulsed out
// on rx_byte/rx_byte_valid one cycle later.
// ---------------------------------------------------------------------------
module rx_byte_interface (
    input  logic       clk,
    input  logic       rst_n,

    input  logic [7:0] host_data,
    input  logic       host_valid,

    output logic       host_ready,

    output logic [7:0] rx_byte,
    output logic       rx_byte_valid
);

    assign host_ready = 1'b1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rx_byte       <= '0;
            rx_byte_valid <= 1'b0;
        end else begin
            rx_byte_valid <= 1'b0;
            if (host_valid && host_ready) begin
                rx_byte       <= host_data;
                rx_byte_valid <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
