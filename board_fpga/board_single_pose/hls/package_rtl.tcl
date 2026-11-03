# Package generated RTL without the old HLS date-dependent IP exporter.
# vivado -mode batch -source package_rtl.tcl -tclargs HLS_SOLUTION OUTPUT
if {$argc < 2 || $argc > 4} {error "Expected HLS solution and separate output directory ?package-only? ?top?"}
set top pose_gemm
if {$argc == 4} {set top [lindex $argv 3]}
set solution [file normalize [lindex $argv 0]]
set output [file normalize [lindex $argv 1]]
set rtl [glob $solution/syn/verilog/*.v]
if {![llength $rtl]} {error "HLS RTL not found"}
file mkdir $output
create_project pose_ip $output/project -part xc7z020clg400-2 -force
add_files $rtl
set_property top $top [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir $output/ip -vendor fpga2026.org -library pose -taxonomy /UserIP -import_files -set_current true
set core [ipx::current_core]
set_property name pose_gemm $core
set_property version 1.0 $core
set_property display_name {Single pose INT8 GEMM} $core
set_property core_revision 1 $core
ipx::infer_bus_interfaces xilinx.com:interface:aximm_rtl:1.0 $core
ipx::infer_bus_interfaces xilinx.com:signal:clock_rtl:1.0 $core
ipx::infer_bus_interfaces xilinx.com:signal:reset_rtl:1.0 $core
set clock [ipx::get_bus_interfaces ap_clk -of_objects $core]
if {[llength $clock]} {
    set param [ipx::add_bus_parameter ASSOCIATED_BUSIF $clock]
    set_property value {s_axi_control:m_axi_weights:m_axi_features:m_axi_results} $param
    set param [ipx::add_bus_parameter ASSOCIATED_RESET $clock]
    set_property value ap_rst_n $param
}
set map [ipx::add_memory_map s_axi_control $core]
set block [ipx::add_address_block regs $map]
set_property range 4096 $block
set_property width 32 $block
set_property slave_memory_map_ref s_axi_control [ipx::get_bus_interfaces s_axi_control -of_objects $core]
foreach name {m_axi_weights m_axi_features m_axi_results} {
    set space [ipx::add_address_space $name $core]
    set_property range 4294967296 $space
    set_property width 32 $space
    set_property master_address_space_ref $name [ipx::get_bus_interfaces $name -of_objects $core]
}
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -quiet $core
ipx::save_core $core
if {$argc >= 3 && [lindex $argv 2] eq "package-only"} {
    puts "POSE_IP_PACKAGED: $output/ip"
    exit
}
synth_design -top $top -part xc7z020clg400-2 -mode out_of_context
create_clock -name ap_clk -period 10.000 [get_ports ap_clk]
opt_design
place_design
route_design
report_utilization -file $output/utilization.txt
report_timing_summary -file $output/timing.txt
report_drc -file $output/drc.txt
write_checkpoint -force $output/pose_gemm_ooc.dcp
set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
if {![llength $setup] || ![llength $hold] || [get_property SLACK $setup]<0 || [get_property SLACK $hold]<0} {error "OOC timing failed"}
puts "POSE_IP_OOC_OK: $output/ip"
exit
