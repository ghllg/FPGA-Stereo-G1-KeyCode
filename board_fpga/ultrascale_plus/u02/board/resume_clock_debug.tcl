set root [file dirname [file normalize [info script]]]
cd $root
set_param general.maxThreads 4
open_project [file join $root debug_project/u02_compute.xpr]
open_bd_design [get_files compute.bd]
generate_target all [get_files compute.bd]
reset_run compute_clock_probe_0_synth_1
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Debug synthesis failed"}
open_run synth_1
puts "U02_DEBUG_HUBS [get_debug_cores]"
set hub [get_debug_cores dbg_hub]
if {[llength $hub] != 1} {error "Debug hub not found"}
set net [get_nets -hier -filter {NAME =~ "*clock_probe*debug_clk*"}]
puts "U02_CFG_DEBUG_CLOCK_NETS $net"
if {![llength $net]} {error "Independent debug clock missing"}
set_property C_CLK_INPUT_FREQ_HZ 66666667 $hub
set_property C_ENABLE_CLK_DIVIDER false $hub
connect_debug_port dbg_hub/clk [lindex $net 0]
write_checkpoint -force [file join $root debug_project/u02_compute.runs/synth_1/compute_wrapper.dcp]
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Debug implementation failed"}
open_run impl_1
set reports [file join $root ../reports/clock_debug]
report_timing_summary -file [file join $reports compute_timing.rpt]
report_utilization -file [file join $reports compute_utilization.rpt]
report_drc -file [file join $reports compute_drc.rpt]
write_debug_probes -force [file join $reports compute.ltx]
puts "U02_INDEPENDENT_CLOCK_DEBUG_BITSTREAM_READY"
close_project
