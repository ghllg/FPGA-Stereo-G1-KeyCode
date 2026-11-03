# Offline candidate only. Launch after source-bound RTL and mapped-capacity gates.
# Reuses the proven PS configuration, POR reset and independent debug clock.
# No board access or boot-medium modification; every output directory must be new.
set inference [file dirname [file normalize [info script]]]
set root [file normalize [file join $inference ../board]]
set repo [file normalize [file join $root ../../..]]
set u02_region_baseline_path [file join $root ps_required_baseline.tsv]
puts "U02_REGION_PS_BASELINE_FILE=$u02_region_baseline_path"
set u02_region_baseline_channel [open $u02_region_baseline_path r]
set u02_region_baseline_text [read $u02_region_baseline_channel]
close $u02_region_baseline_channel
foreach name {U02_REGION_PROJECT_DIR U02_REGION_IP_DIR U02_REGION_IP_VLNV U02_REGION_REPORT_DIR} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {error "Required explicit candidate setting: $name"}
}
set project_dir [file normalize $::env(U02_REGION_PROJECT_DIR)]
set ip_dir [file normalize $::env(U02_REGION_IP_DIR)]
set ip_vlnv $::env(U02_REGION_IP_VLNV)
set reports [file normalize $::env(U02_REGION_REPORT_DIR)]
foreach path [list $project_dir $reports] {
    if {[file exists $path]} {error "Fresh output directory required: $path"}
    if {![regexp -nocase {^[DE]:/} $path]} {error "Candidate output must be on D/E: $path"}
}
if {![file isfile [file join $ip_dir component.xml]]} {error "Exported candidate component.xml missing"}
file mkdir $reports
cd $inference
set_param general.maxThreads 2
package require msgcat
create_project u02_region_dual $project_dir -part xczu5ev-sfvc784-2-i
set_property target_language Verilog [current_project]
set_property ip_repo_paths [list $ip_dir] [current_project]
update_ip_catalog
create_bd_design compute
add_files [file join $root por_reset.v]
create_bd_cell -type module -reference u02_por_reset por
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.2 ps
source [file join $root ps_config_2018.tcl]
configure_u02_ps ps $repo
foreach line [split [string trim $u02_region_baseline_text] "\n"] {
    set line [string trimright $line "\r"]
    if {$line eq ""} {continue}
    lassign [split $line "\t"] key expected
    if {[get_property CONFIG.$key [get_bd_cells ps]] ne $expected} {error "PS baseline differs: $key"}
}
create_bd_cell -type ip -vlnv $ip_vlnv compute_ip
foreach port {m_axi_commands m_axi_weights m_axi_features s_axi_control} {
    if {[llength [get_bd_intf_pins -quiet compute_ip/$port]] != 1} {error "Candidate interface missing: $port"}
}
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
foreach {port index} {commands 0 weights 1 features 2} {
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
foreach port {commands weights features} {
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
generate_target synthesis [get_files compute.bd]
add_files -norecurse [make_wrapper -files [get_files compute.bd] -top]
set_property top compute_wrapper [current_fileset]
write_bd_tcl [file join $reports compute_bd.tcl]
launch_runs synth_1 -jobs 1
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Region synthesis failed"}
open_run synth_1
set hub [get_debug_cores dbg_hub]
set net [get_nets -hier -filter {NAME =~ "*clock_probe*debug_clk*"}]
if {[llength $hub] != 1 || ![llength $net]} {error "Independent debug clock missing"}
set_property C_CLK_INPUT_FREQ_HZ 66666667 $hub
set_property C_ENABLE_CLK_DIVIDER false $hub
connect_debug_port dbg_hub/clk [lindex $net 0]
write_checkpoint -force [file join $project_dir u02_region_dual.runs/synth_1/compute_wrapper.dcp]
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Region implementation failed"}
open_run impl_1
report_timing_summary -file [file join $reports timing.rpt]
report_utilization -file [file join $reports utilization.rpt]
report_drc -file [file join $reports drc.rpt]
write_debug_probes -force [file join $reports compute.ltx]
puts "U02_REGION_DUAL_CANDIDATE_BIT_READY_NOT_DEPLOYED"
close_project
