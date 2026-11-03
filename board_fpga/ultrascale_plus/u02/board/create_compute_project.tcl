# Standalone AXU5EVB-E PS/PL design. Never open/downgrade a vendor project.
set root [file dirname [file normalize [info script]]]
set repo [file normalize [file join $root ../../..]]
cd $root
set_param general.maxThreads 4
set project_directory compute_project
set report_directory [file normalize [file join $root ../reports]]
set debug_mode [expr {[llength $argv] && [lindex $argv 0] eq "clock_debug"}]
if {$debug_mode} {
    set project_directory debug_project
    set report_directory [file join $report_directory clock_debug]
    file mkdir $report_directory
}
create_project u02_compute [file join $root $project_directory] -part xczu5ev-sfvc784-2-i
set_property ip_repo_paths [list [file join $root hls/ddr_compute_project/zu5ev/impl/ip]] [current_project]
update_ip_catalog
create_bd_design compute
add_files [file join $root por_reset.v]
create_bd_cell -type module -reference u02_por_reset por
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.2 ps]
source [file join $root ps_config_2018.tcl]
configure_u02_ps ps $repo

# HWH values are comparison evidence, not a cross-version project import.
set baseline [open [file join $root ps_required_baseline.tsv] r]
while {[gets $baseline line] >= 0} {
    set fields [split $line "\t"]
    set key [lindex $fields 0]
    set expected [lindex $fields 1]
    set actual [get_property CONFIG.$key $ps]
    if {$actual ne $expected} {error "PS baseline mismatch $key expected=$expected actual=$actual"}
}
close $baseline
puts "U02_PS_DDR_MIO_CLOCK_BASELINE_MATCHED"

create_bd_cell -type ip -vlnv u02:platform:u02_ddr_compute:1.0 compute_ip
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 control
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ddr
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells control]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells ddr]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset
set_property CONFIG.C_EXT_RESET_HIGH {0} [get_bd_cells reset]
set_property CONFIG.C_AUX_RESET_HIGH {1} [get_bd_cells reset]
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 locked
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] [get_bd_cells locked]
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 zero
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {0}] [get_bd_cells zero]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins control/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control/M00_AXI] [get_bd_intf_pins compute_ip/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins compute_ip/m_axi_memory] [get_bd_intf_pins ddr/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ddr/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
foreach pin {ps/maxihpm0_fpd_aclk ps/maxihpm0_lpd_aclk ps/saxihp0_fpd_aclk compute_ip/ap_clk reset/slowest_sync_clk por/clk control/ACLK control/S00_ACLK control/M00_ACLK ddr/ACLK ddr/S00_ACLK ddr/M00_ACLK} {
    connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins $pin]
}
connect_bd_net [get_bd_pins por/reset_n] [get_bd_pins reset/ext_reset_in]
connect_bd_net [get_bd_pins locked/dout] [get_bd_pins reset/dcm_locked]
connect_bd_net [get_bd_pins zero/dout] [get_bd_pins reset/aux_reset_in] [get_bd_pins reset/mb_debug_sys_rst]
foreach pin {compute_ip/ap_rst_n control/ARESETN control/S00_ARESETN control/M00_ARESETN ddr/ARESETN ddr/S00_ARESETN ddr/M00_ARESETN} {
    connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins $pin]
}
connect_bd_net [get_bd_pins compute_ip/interrupt] [get_bd_pins ps/pl_ps_irq0]
if {$debug_mode} {source [file join $root add_clock_probe.tcl]}
assign_bd_address
foreach slave {ps/SAXIGP2/HP0_PCIE_LOW ps/SAXIGP2/HP0_QSPI} {
    exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces compute_ip/Data_m_axi_memory] [get_bd_addr_segs $slave]
}
set control_seg [get_bd_addr_segs ps/Data/SEG_compute_ip_Reg]
set_property range 64K $control_seg
set_property offset 0xA0000000 $control_seg
validate_bd_design
save_bd_design
set output [open [file join $report_directory ps_compute.tsv] w]
foreach property [lsort [list_property $ps]] {
    if {[string match CONFIG.* $property]} {
        puts $output "[string range $property 7 end]\t[get_property $property $ps]"
    }
}
close $output
set output [open [file join $report_directory compute_address.tsv] w]
foreach space [get_bd_addr_spaces] {
    foreach segment [get_bd_addr_segs -of_objects $space] {
        puts $output "$space\t$segment\t[get_property OFFSET $segment]\t[get_property RANGE $segment]"
    }
}
close $output
generate_target all [get_files compute.bd]
set wrapper [make_wrapper -files [get_files compute.bd] -top]
add_files -norecurse $wrapper
set_property top compute_wrapper [current_fileset]
write_bd_tcl [file join $report_directory compute_bd_export.tcl]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis failed"}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation/bitstream failed"}
open_run impl_1
report_timing_summary -file [file join $report_directory compute_timing.rpt]
report_utilization -file [file join $report_directory compute_utilization.rpt]
report_drc -file [file join $report_directory compute_drc.rpt]
write_hwdef -file [file join $report_directory compute.hwdef]
if {$debug_mode} {write_debug_probes -force [file join $report_directory compute.ltx]}
puts "U02_COMPUTE_BITSTREAM_READY"
close_project
