set root [file dirname [file normalize [info script]]]
set generated [file normalize $::env(U02_REGION_GENERATED_DIR)]
set project [file normalize $::env(U02_REGION_HLS_PROJECT)]
if {[file exists $project]} {error "Fresh isolated dual-mode project required"}
cd [file dirname $project]
open_project [file tail $project]
set_top u02_region_dual
set flags "-std=c++11 -ffp-contract=off -I$generated -I$root"
add_files [file join $root stream_region_dual.cpp] -cflags $flags
add_files -tb [file join $root test_region_dual.cpp] -cflags $flags
open_solution zu5ev
set_part {xczu5ev-sfvc784-2-i}
create_clock -period 10.312 -name default
config_interface -m_axi_addr64
csim_design -O
if {[info exists ::env(U02_REGION_SYNTHESIS)] && $::env(U02_REGION_SYNTHESIS) eq "1"} {csynth_design}
exit
