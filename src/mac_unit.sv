// ---------------------------------------------------------------------------
// mac_unit -- output-stationary multiply-accumulate cell.
//
// This is the only module that interprets the number format. The rest of
// the array carries operands as opaque bit vectors, so a different MAC
// (e.g. floating point) can replace this one with the same port list.
//
// Contract:
//   - combinational multiply, registered accumulate (1 result/cycle, no stall)
//   - clr && en  : acc <= 0 + a*b        (start a new dot product)
//   - en         : acc <= acc + a*b
//   - clr only   : acc <= 0
//   - !en && !clr: hold
// ---------------------------------------------------------------------------
module mac_unit #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                clr,    // zero the accumulator this cycle
    input  logic                en,     // accumulate this cycle
    input  logic [DATA_W-1:0]   a,
    input  logic [DATA_W-1:0]   b,
    output logic [ACC_W-1:0]    acc
);

    logic signed [2*DATA_W-1:0] prod;
    assign prod = $signed(a) * $signed(b);

    logic signed [ACC_W-1:0] acc_q;

    always_ff @(posedge clk) begin
        if (!rst_n)     acc_q <= '0;
        else if (en)    acc_q <= (clr ? '0 : acc_q) + ACC_W'(prod);
        else if (clr)   acc_q <= '0;
    end

    assign acc = acc_q;

endmodule
