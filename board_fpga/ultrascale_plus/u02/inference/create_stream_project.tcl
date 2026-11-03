# Independent whole-network backend project using the proven U02 PS/clock/reset paths.
set inference [file dirname [file normalize [info script]]]
set root [file normalize [file join $inference ../board]]
set repo [file normalize [file join $root ../../..]]
set reports [file normalize [file join $inference ../reports/stream_implementation]]
set project_dir [file join $inference stream_project]
set hls_project [file join $inference stream_hls_03]
if {[info exists ::env(U02_STREAM_PROJECT_DIR)]} {set project_dir [file normalize $::env(U02_STREAM_PROJECT_DIR)]}
if {[info exists ::env(U02_STREAM_HLS_PROJECT)]} {set hls_project [file normalize $::env(U02_STREAM_HLS_PROJECT)]}
if {[info exists ::env(U02_STREAM_REPORT_DIR)]} {set reports [file normalize $::env(U02_STREAM_REPORT_DIR)]}
file mkdir $reports
cd $inference
set_param general.maxThreads 2
if {[info exists ::env(U02_STREAM_RESUME_EMPTY)] && $::env(U02_STREAM_RESUME_EMPTY) eq "1"} {
    # Explicit recovery only for the archived empty project. Never resume runs.
    open_project [file join $project_dir u02_stream.xpr]
    if {[get_property PART [current_project]] ne "xczu5ev-sfvc784-2-i"} {error "Unexpected recovery part"}
    set_property ip_repo_paths [list [file join $hls_project zu5ev/impl/ip]] [current_project]
    update_ip_catalog
    open_bd_design [file join $project_dir u02_stream.srcs sources_1 bd compute compute.bd]
    if {[llength [get_bd_cells -quiet]]} {error "Recovery requires an empty block design"}
    set xgui [file join $project_dir u02_stream.srcs sources_1 bd mref u02_por_reset xgui u02_por_reset_v1_0.tcl]
    set fd [open $xgui r]; set gui_text [read $fd]; close $fd
    if {[string length $gui_text] < 100} {error "Generated module XGUI is incomplete"}
    if {![llength [get_files -quiet [file join $root por_reset.v]]]} {add_files [file join $root por_reset.v]}
    puts U02_EMPTY_PROJECT_GENERATED_XGUI_RECOVERY_PREFLIGHT_PASSED
} else {
    if {[file exists $project_dir]} {error "A fresh project directory is required; preserve existing builds"}
    create_project u02_stream $project_dir -part xczu5ev-sfvc784-2-i
    set_property target_language Verilog [current_project]
    set_property ip_repo_paths [list [file join $hls_project zu5ev/impl/ip]] [current_project]
    update_ip_catalog
    create_bd_design compute
    add_files [file join $root por_reset.v]
}
create_bd_cell -type module -reference u02_por_reset por
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.2 ps
source [file join $root ps_config_2018.tcl]
configure_u02_ps ps $repo
set baseline [open [file join $root ps_required_baseline.tsv] r]
while {[gets $baseline line] >= 0} {
    lassign [split $line "\t"] key expected
    if {[get_property CONFIG.$key [get_bd_cells ps]] ne $expected} {error "PS baseline differs: $key"}
}
close $baseline
create_bd_cell -type ip -vlnv u02:inference:u02_stream:1.0 compute_ip
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 control
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ddr
set_property -dict [list CONFIG.NUM_SI 2 CONFIG.NUM_MI 1] [get_bd_cells control]
set_property -dict [list CONFIG.NUM_SI 3 CONFIG.NUM_MI 1] [get_bd_cells ddr]
create_bd_cell -type ip -vlnv xilinx.com:ip:jtag_axi:1.2 jtag_axi
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset
set_property -dict [list CONFIG.C_EXT_RESET_HIGH 0 CONFIG.C_AUX_RESET_HIGH 1] [get_bd_cells reset]
foreach {name value} {locked 1 zero 0} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 $name
    set_property -dict [list CONFIG.CONST_WIDTH 1 CONFIG.CONST_VAL $value] [get_bd_cells $name]
}
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins control/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins jtag_axi/M_AXI] [get_bd_intf_pins control/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins control/M00_AXI] [get_bd_intf_pins compute_ip/s_axi_control]
foreach {port index} {weights 0 features 1 results 2} {
    connect_bd_intf_net [get_bd_intf_pins compute_ip/m_axi_$port] [get_bd_intf_pins ddr/S0${index}_AXI]
}
connect_bd_intf_net [get_bd_intf_pins ddr/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
foreach pin {ps/maxihpm0_fpd_aclk ps/maxihpm0_lpd_aclk ps/saxihp0_fpd_aclk compute_ip/ap_clk reset/slowest_sync_clk por/clk jtag_axi/aclk control/ACLK control/S00_ACLK control/S01_ACLK control/M00_ACLK ddr/ACLK ddr/S00_ACLK ddr/S01_ACLK ddr/S02_ACLK ddr/M00_ACLK} {
    connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins $pin]
}
connect_bd_net [get_bd_pins por/reset_n] [get_bd_pins reset/ext_reset_in]
connect_bd_net [get_bd_pins locked/dout] [get_bd_pins reset/dcm_locked]
connect_bd_net [get_bd_pins zero/dout] [get_bd_pins reset/aux_reset_in] [get_bd_pins reset/mb_debug_sys_rst]
foreach pin {compute_ip/ap_rst_n jtag_axi/aresetn control/ARESETN control/S00_ARESETN control/S01_ARESETN control/M00_ARESETN ddr/ARESETN ddr/S00_ARESETN ddr/S01_ARESETN ddr/S02_ARESETN ddr/M00_ARESETN} {
    connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins $pin]
}
connect_bd_net [get_bd_pins compute_ip/interrupt] [get_bd_pins ps/pl_ps_irq0]
source [file join $root add_clock_probe.tcl]
assign_bd_address
foreach port {weights features results} {
    foreach slave {ps/SAXIGP2/HP0_PCIE_LOW ps/SAXIGP2/HP0_QSPI} {
        exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces compute_ip/Data_m_axi_$port] [get_bd_addr_segs $slave]
    }
}
foreach space {ps/Data jtag_axi/Data} {
    set seg [get_bd_addr_segs $space/SEG_compute_ip_Reg]
    set_property range 64K $seg
    set_property offset 0xA0000000 $seg
}
validate_bd_design
save_bd_design
generate_target all [get_files compute.bd]
add_files -norecurse [make_wrapper -files [get_files compute.bd] -top]
set_property top compute_wrapper [current_fileset]
write_bd_tcl [file join $reports compute_bd.tcl]
launch_runs synth_1 -jobs 1
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Streaming synthesis failed"}
open_run synth_1
set hub [get_debug_cores dbg_hub]
set net [get_nets -hier -filter {NAME =~ "*clock_probe*debug_clk*"}]
if {[llength $hub] != 1 || ![llength $net]} {error "Independent debug clock missing"}
set_property C_CLK_INPUT_FREQ_HZ 66666667 $hub
set_property C_ENABLE_CLK_DIVIDER false $hub
connect_debug_port dbg_hub/clk [lindex $net 0]
write_checkpoint -force [file join $project_dir u02_stream.runs/synth_1/compute_wrapper.dcp]
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Streaming implementation failed"}
open_run impl_1
report_timing_summary -file [file join $reports timing.rpt]
report_utilization -file [file join $reports utilization.rpt]
report_drc -file [file join $reports drc.rpt]
write_debug_probes -force [file join $reports compute.ltx]
puts "U02_STREAM_FULL_NETWORK_BACKEND_BIT_READY"
close_project
