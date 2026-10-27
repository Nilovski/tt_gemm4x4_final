// ---------------------------------------------------------------------------
// systolic_array -- N x N output-stationary mesh.
//
// A enters from the WEST edge and flows east; B enters from the NORTH edge and
// flows south. PE(i,j) accumulates C[i][j] in place -- nothing moves at the
// output side, so results are read through a mux rather than a drain chain.
// ---------------------------------------------------------------------------
module systolic_array #(
    parameter int N      = 4,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clr,
    input  logic                    en,
    input  logic [N*DATA_W-1:0]     a_west,   // a_west[i]  -> row i
    input  logic [N*DATA_W-1:0]     b_north,  // b_north[j] -> column j
    output logic [N*N*ACC_W-1:0]    c_flat    // c_flat[i*N+j] = C[i][j]
);

    logic [DATA_W-1:0] a_h [N][N+1];  // horizontal (west -> east) links
    logic [DATA_W-1:0] b_v [N+1][N];  // vertical   (north -> south) links
    logic [ACC_W-1:0]  acc [N][N];

    // Edge injection
    for (genvar i = 0; i < N; i++) begin : g_west_edge
        assign a_h[i][0] = a_west[i*DATA_W +: DATA_W];
    end
    for (genvar j = 0; j < N; j++) begin : g_north_edge
        assign b_v[0][j] = b_north[j*DATA_W +: DATA_W];
    end

    // The mesh
    for (genvar i = 0; i < N; i++) begin : g_row
        for (genvar j = 0; j < N; j++) begin : g_col
            pe #(
                .DATA_W (DATA_W),
                .ACC_W  (ACC_W)
            ) u_pe (
                .clk   (clk),
                .rst_n (rst_n),
                .clr   (clr),
                .en    (en),
                .a_in  (a_h[i][j]),
                .b_in  (b_v[i][j]),
                .a_out (a_h[i][j+1]),
                .b_out (b_v[i+1][j]),
                .acc   (acc[i][j])
            );
            assign c_flat[(i*N + j)*ACC_W +: ACC_W] = acc[i][j];
        end
    end

endmodule
