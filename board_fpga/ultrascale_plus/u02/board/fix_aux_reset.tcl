# Fix the observed permanently asserted auxiliary reset and add a JTAG AXI probe.
set root [file dirname [file normalize [info script]]]
cd $root
set_param general.maxThreads 4
open_project [file join $root debug_project/u02_compute.xpr]
open_bd_design [get_files compute.bd]
set_property CONFIG.C_AUX_RESET_HIGH {1} [get_bd_cells reset]
if {[get_property CONFIG.C_EXT_RESET_HIGH [get_bd_cells reset]] ne "0" ||
    [get_property CONFIG.C_AUX_RESET_HIGH [get_bd_cells reset]] ne "1"} {error "Reset polarities did not apply"}
create_bd_cell -type ip -vlnv xilinx.com:ip:jtag_axi:1.2 jtag_axi
set_property CONFIG.NUM_SI {2} [get_bd_cells control]
connect_bd_intf_net [get_bd_intf_pins jtag_axi/M_AXI] [get_bd_intf_pins control/S01_AXI]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins jtag_axi/aclk] [get_bd_pins control/S01_ACLK]
connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins jtag_axi/aresetn] [get_bd_pins control/S01_ARESETN]
assign_bd_address -offset 0xA0000000 -range 64K -target_address_space [get_bd_addr_spaces jtag_axi/Data] [get_bd_addr_segs compute_ip/s_axi_control/Reg]
validate_bd_design
save_bd_design
generate_target all [get_files compute.bd]
set reports [file join $root ../reports/reset_fixed]
file mkdir $reports
write_bd_tcl [file join $reports compute_bd.tcl]
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Corrected reset synthesis failed"}
open_run synth_1
set hub [get_debug_cores dbg_hub]
set net [get_nets -hier -filter {NAME =~ "*clock_probe*debug_clk*"}]
if {[llength $hub] != 1 || ![llength $net]} {error "Independent debug clock missing"}
set_property C_CLK_INPUT_FREQ_HZ 66666667 $hub
set_property C_ENABLE_CLK_DIVIDER false $hub
connect_debug_port dbg_hub/clk [lindex $net 0]
write_checkpoint -force [file join $root debug_project/u02_compute.runs/synth_1/compute_wrapper.dcp]
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Corrected reset implementation failed"}
open_run impl_1
report_timing_summary -file [file join $reports compute_timing.rpt]
report_utilization -file [file join $reports compute_utilization.rpt]
report_drc -file [file join $reports compute_drc.rpt]
write_debug_probes -force [file join $reports compute.ltx]
puts "U02_RESET_POLARITY_AND_JTAG_AXI_BIT_READY"
close_project
