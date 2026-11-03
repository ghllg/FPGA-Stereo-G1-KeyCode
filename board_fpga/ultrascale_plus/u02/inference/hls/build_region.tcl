set root [file dirname [file normalize [info script]]]
set project [file normalize $::env(U02_REGION_HLS_PROJECT)]
set generated [file normalize $::env(U02_REGION_GENERATED_DIR)]
if {[file exists $project]} {error "Fresh isolated region HLS project required"}
cd [file dirname $project]
open_project [file tail $project]
set_top u02_region
set flags "-std=c++11 -ffp-contract=off -I$generated -I$root"
add_files [file join $root stream_region_top.cpp] -cflags $flags
add_files -tb [file join $root .. host test_stream_region_execute.cpp] -cflags "$flags -DU02_REGION_HLS_TEST"
open_solution zu5ev
set_part {xczu5ev-sfvc784-2-i}
create_clock -period 10.312 -name default
config_interface -m_axi_addr64
set args "$::env(U02_REGION_METADATA) $::env(U02_REGION_MODEL) $::env(U02_FP32_FIXTURE_DIR) $generated"
csim_design -O -argv $args
if {[info exists ::env(U02_REGION_SYNTHESIS)] && $::env(U02_REGION_SYNTHESIS) eq "1"} {csynth_design}
exit
