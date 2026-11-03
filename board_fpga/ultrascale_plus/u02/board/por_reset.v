// Deterministic FPGA startup reset, driven by the existing PS PL0 clock.
module u02_por_reset(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *) input wire clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 reset_n RST",
       X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *) output wire reset_n
);
    reg [5:0] count = 6'b000000;
    always @(posedge clk) begin
        if (count != 6'b111111) count <= count + 1'b1;
    end
    assign reset_n = (count == 6'b111111);
endmodule
