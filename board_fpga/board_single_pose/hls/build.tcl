# Vivado HLS 2018.3. Does not program the board or touch the existing BD.
set root [file dirname [file normalize [info script]]]
if {![info exists ::env(POSE_HLS_BUILD)]} {error "Set POSE_HLS_BUILD to a separate build directory"}
set dest [file normalize $::env(POSE_HLS_BUILD)]
if {$dest eq $root || [string first "$root/" "$dest/"] == 0} {error "Use a separate build directory"}
file mkdir [file dirname $dest]
cd [file dirname $dest]
open_project [file tail $dest]
set top pose_gemm
if {[info exists ::env(POSE_HLS_TOP)]} {set top $::env(POSE_HLS_TOP)}
set_top $top
add_files $root/$top.cpp
if {$top eq "pose_conv_narrow"} {
    add_files $root/pose_quantize_stream.cpp -cflags "-DPOSE_QUANT_SUBFUNCTION"
    add_files -tb $root/test_conv_quant.cpp -cflags "-DPOSE_TEST_NARROW_CONV"
} elseif {$top eq "pose_conv_flat_dsp"} {
    add_files $root/pose_quantize_stream.cpp -cflags "-DPOSE_QUANT_SUBFUNCTION"
    add_files -tb $root/test_conv_quant.cpp -cflags "-DPOSE_TEST_FLAT_DSP_CONV"
} elseif {$top eq "pose_conv_dsp"} {
    add_files $root/pose_quantize_stream.cpp -cflags "-DPOSE_QUANT_SUBFUNCTION"
    add_files -tb $root/test_conv_quant.cpp -cflags "-DPOSE_TEST_DSP_CONV"
} elseif {$top eq "pose_conv_burst"} {
    add_files $root/pose_quantize_burst.cpp -cflags "-DPOSE_QUANT_SUBFUNCTION"
    add_files -tb $root/test_conv_quant.cpp -cflags "-DPOSE_TEST_BURST_CONV"
} elseif {$top eq "pose_conv_flat"} {
    add_files $root/pose_quantize_stream.cpp -cflags "-DPOSE_QUANT_SUBFUNCTION"
    add_files -tb $root/test_conv_quant.cpp -cflags "-DPOSE_TEST_FLAT_CONV"
} elseif {$top eq "pose_conv_wide"} {
    add_files $root/pose_quantize_stream.cpp -cflags "-DPOSE_QUANT_SUBFUNCTION"
    add_files -tb $root/test_conv_quant.cpp -cflags "-DPOSE_TEST_WIDE_CONV"
} elseif {$top eq "pose_conv_quant"} {
    add_files $root/pose_quantize_stream.cpp -cflags "-DPOSE_QUANT_SUBFUNCTION"
    add_files -tb $root/test_conv_quant.cpp
} elseif {$top eq "pose_quantize_stream"} {
    add_files -tb $root/test_quantize_stream.cpp
} elseif {$top eq "pose_conv_packed"} {
    add_files -tb $root/test_conv.cpp -cflags "-DPOSE_TEST_PACKED"
} elseif {$top eq "pose_conv_reuse"} {
    add_files -tb $root/test_conv.cpp
} elseif {$top eq "pose_gemm_reuse_pipe"} {
    add_files -tb $root/test_gemm.cpp -cflags "-DPOSE_TEST_WIDE -DPOSE_TEST_REUSE -DPOSE_TEST_REUSE_PIPE"
} elseif {$top eq "pose_gemm_reuse"} {
    add_files -tb $root/test_gemm.cpp -cflags "-DPOSE_TEST_WIDE -DPOSE_TEST_REUSE"
} elseif {$top eq "pose_gemm_wide"} {
    add_files -tb $root/test_gemm.cpp -cflags "-DPOSE_TEST_WIDE"
} else {add_files -tb $root/test_gemm.cpp}
open_solution solution1
set_part xc7z020clg400-2
set period 6.667
if {[info exists ::env(POSE_HLS_PERIOD_NS)]} {set period $::env(POSE_HLS_PERIOD_NS)}
create_clock -period $period -name default
csim_design
csynth_design
if {[info exists ::env(POSE_HLS_EXPORT)] && $::env(POSE_HLS_EXPORT) eq "1"} {
    export_design -format ip_catalog -vendor fpga2026.org -library pose -version 1.0
}
exit
