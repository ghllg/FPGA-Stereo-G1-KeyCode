set root [file dirname [file normalize [info script]]]
set project [file normalize $::env(U02_DW128_PROJECT)]
if {[file exists $project]} {error "Fresh isolated DW128 v2 project required"}
cd [file dirname $project]
open_project [file tail $project]
set_top u02_stream_128write_v2
add_files [file join $root stream_dw128_write_v2.cpp] -cflags "-std=c++11 -ffp-contract=off"
add_files -tb [file join $root test_stream_dw128_write_v2.cpp] -cflags "-std=c++11 -ffp-contract=off"
open_solution zu5ev
set_part {xczu5ev-sfvc784-2-i}
create_clock -period 10.312 -name default
config_interface -m_axi_addr64
csim_design -O
csynth_design
exit
