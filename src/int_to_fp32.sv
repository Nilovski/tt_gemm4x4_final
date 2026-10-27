`default_nettype none

// ---------------------------------------------------------------------------
// int_to_fp32 -- y = x * 2^scale as an IEEE-754 single.
//
// x is a signed accumulator of up to 24 bits, so the conversion is exact
// whenever the result is in the normal range. Results below the smallest
// normal flush to signed zero; results above the largest finite value
// become +/-inf.
// ---------------------------------------------------------------------------
module int_to_fp32 #(
    parameter int XW = 21
) (
    input  logic signed [XW-1:0] x,
    input  logic signed [9:0]    scale,
    output logic        [31:0]   y
);

    localparam int PW = $clog2(XW + 1);

    logic          s;
    logic [XW-1:0] mag;
    logic [PW-1:0] msb;                       // index of the leading one
    logic signed [11:0] e;                    // biased exponent
    logic [XW-1:0] norm;

    assign s   = x[XW-1];
    assign mag = s ? XW'(-x) : XW'(x);

    always_comb begin
        msb = '0;
        for (int i = 0; i < XW; i++)
            if (mag[i]) msb = PW'(i);
    end

    // value = mag * 2^scale = 1.f * 2^(msb + scale)
    assign e    = 12'sd127 + 12'(scale) + $signed({1'b0, 11'(msb)});
    assign norm = mag << (PW'(XW - 1) - msb);  // leading one at bit XW-1

    always_comb begin
        if (mag == '0 || e <= 12'sd0) y = {s, 31'd0};
        else if (e >= 12'sd255)       y = {s, 8'hFF, 23'd0};
        else                          y = {s, e[7:0], 23'(norm[XW-2:0]) << (23 - (XW - 1))};
    end

    wire _unused = &{norm[XW-1], 1'b0};   // the leading one is implicit

endmodule

`default_nettype wire
