set root [file dirname [file normalize [info script]]]
set project [file normalize $::env(U02_STREAM_HLS_PROJECT)]
if {[file exists $project]} {error "Fresh streaming build required"}
cd [file dirname $project]
open_project [file tail $project]
set_top u02_stream
add_files [file join $root stream_core.cpp] -cflags "-std=c++11 -ffp-contract=off"
add_files -tb [file join $root test_stream.cpp] -cflags "-std=c++11 -ffp-contract=off"
open_solution zu5ev
set_part {xczu5ev-sfvc784-2-i}
create_clock -period 10.312 -name default
config_interface -m_axi_addr64
csim_design -O
if {[info exists ::env(U02_STREAM_SYNTHESIS)] && $::env(U02_STREAM_SYNTHESIS) eq "1"} {csynth_design}
exit
