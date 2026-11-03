# Continue the existing U02 design from validated generated IP caches.
set root [file dirname [file normalize [info script]]]
cd $root
set_param general.maxThreads 4
open_project [file join $root compute_project/u02_compute.xpr]
open_bd_design [get_files compute.bd]
add_files [file join $root por_reset.v]
create_bd_cell -type module -reference u02_por_reset por
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins por/clk]
disconnect_bd_net [get_bd_nets -of_objects [get_bd_pins reset/ext_reset_in]] [get_bd_pins reset/ext_reset_in]
connect_bd_net [get_bd_pins por/reset_n] [get_bd_pins reset/ext_reset_in]
validate_bd_design
save_bd_design
generate_target all [get_files compute.bd]
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis failed"}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}
open_run impl_1
report_timing_summary -file [file join $root ../reports/compute_por_timing.rpt]
report_utilization -file [file join $root ../reports/compute_por_utilization.rpt]
report_drc -file [file join $root ../reports/compute_por_drc.rpt]
write_hwdef -file [file join $root ../reports/compute_por.hwdef]
write_bd_tcl [file join $root ../reports/compute_por_bd_export.tcl]
puts "U02_COMPUTE_POR_BITSTREAM_READY"
close_project
