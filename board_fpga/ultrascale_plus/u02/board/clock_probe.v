// Observe PS PL0 clock and resets without making PS-side AXI accesses.
// A configuration oscillator drives the debug hub even if PL0 is stopped.
module u02_clock_probe(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ps_clk CLK" *) input wire ps_clk,
    input wire por_n,
    input wire fabric_reset_n,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 debug_clk CLK",
       X_INTERFACE_PARAMETER = "FREQ_HZ 66666667" *) output wire debug_clk,
    output wire [31:0] debug_count,
    output wire [31:0] pl_gray,
    output wire [1:0] resets
);
    wire cfg_clk;
    STARTUPE3 startup_cfg(.CFGMCLK(cfg_clk), .DO(4'b0000), .DTS(4'b1111),
        .FCSBO(1'b1), .FCSBTS(1'b1), .GSR(1'b0),
        .GTS(1'b0), .KEYCLEARB(1'b1), .PACK(1'b0), .USRCCLKO(1'b0),
        .USRCCLKTS(1'b1), .USRDONEO(1'b1), .USRDONETS(1'b1));
    BUFG debug_buffer(.I(cfg_clk), .O(debug_clk));
    reg [31:0] cfg_count = 0;
    reg [31:0] ps_count = 0;
    // Register Gray code in its source domain before observing it elsewhere.
    reg [31:0] ps_gray = 0;
    wire [31:0] next_ps = ps_count + 1'b1;
    always @(posedge ps_clk) begin
        ps_count <= next_ps;
        ps_gray <= next_ps ^ (next_ps >> 1);
    end
    (* ASYNC_REG = "TRUE" *) reg [31:0] gray_meta = 0, gray_sync = 0;
    (* ASYNC_REG = "TRUE" *) reg [1:0] reset_meta = 0, reset_sync = 0;
    always @(posedge debug_clk) begin
        cfg_count <= cfg_count + 1'b1;
        gray_meta <= ps_gray;
        gray_sync <= gray_meta;
        reset_meta <= {fabric_reset_n, por_n};
        reset_sync <= reset_meta;
    end
    assign debug_count = cfg_count;
    assign pl_gray = gray_sync;
    assign resets = reset_sync;
endmodule
