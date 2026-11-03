# Migrate the existing complete-network convolution backend, not a fixed V13 tile.
set root [file dirname [file normalize [info script]]]
set repo [file normalize [file join $root ../../../..]]
set common [file join $repo board_single_pose/hls]
set light [file join $repo board_light_pose/hls]
set build [file normalize [file join $root ../v06f_hls]]
if {[file exists $build]} {error "Use a fresh U02 build; do not overwrite a previous result"}
cd [file dirname $build]
open_project [file tail $build]
set_top pose_light_conv
add_files [file join $light pose_light_conv.cpp] -cflags "-I$common"
add_files [file join $common pose_quantize_stream.cpp] -cflags "-DPOSE_QUANT_SUBFUNCTION"
add_files -tb [file join $light test_depthwise.cpp] -cflags "-I$common -DLIGHT_TOP_TEST -DLIGHT_TEST_PAIRS"
add_files -tb [file join $light test_pairs.cpp] -cflags "-I$common -DLIGHT_NO_MAIN"
open_solution zu5ev
set_part xczu5ev-sfvc784-2-i
create_clock -period 10.0 -name default
config_interface -m_axi_addr64
csim_design
puts "U02_V06F_CONTRACT_CSIM_PASSED"
csynth_design
puts "U02_V06F_ZU5EV_SYNTHESIS_READY"
exit
