# Reuse the verified full PS/clock/reset/address construction for HLS23.
if {![info exists ::env(U02_FP32_IP_PACKAGE)]} {error "Verified HLS23 package required"}
set u02_partial_base [file join [file dirname [file normalize [info script]]] create_stream_project.tcl]
set u02_partial_channel [open $u02_partial_base r]
set u02_partial_text [read $u02_partial_channel]
close $u02_partial_channel
set u02_partial_needle {[file join $hls_project zu5ev/impl/ip]}
if {[string first $u02_partial_needle $u02_partial_text] < 0} {error "Package binding missing"}
if {[string first {set_param general.maxThreads 2} $u02_partial_text] < 0} {error "Thread setting missing"}
set u02_partial_text [string map [list \
    $u02_partial_needle {[file normalize $::env(U02_FP32_IP_PACKAGE)]} \
    {set_param general.maxThreads 2} {set_param general.maxThreads 1}] $u02_partial_text]
if {![info exists ::env(U02_STREAM_BUILD_PHASE)]} {error "Explicit build phase required"}
set u02_partial_phase $::env(U02_STREAM_BUILD_PHASE)
set u02_partial_impl {launch_runs impl_1 -to_step write_bitstream -jobs 1}
set u02_partial_report {open_run impl_1}
set u02_partial_impl_pos [string first $u02_partial_impl $u02_partial_text]
set u02_partial_report_pos [string first $u02_partial_report $u02_partial_text]
if {$u02_partial_impl_pos < 0 || $u02_partial_report_pos <= $u02_partial_impl_pos} {error "Phase boundaries missing"}
if {$u02_partial_phase eq "synthesis"} {
    # Exit the parent after writing the proven debug-hub synthesis checkpoint.
    # Its retained heap must not coexist with the implementation child process.
    set u02_partial_text [string range $u02_partial_text 0 [expr {$u02_partial_impl_pos - 1}]]
    append u02_partial_text "\nclose_project\nexit\n"
    uplevel #0 $u02_partial_text
} elseif {$u02_partial_phase eq "implementation" || $u02_partial_phase eq "reports"} {
    set project_dir [file normalize $::env(U02_STREAM_PROJECT_DIR)]
    set reports [file normalize $::env(U02_STREAM_REPORT_DIR)]
    set_param general.maxThreads 1
    open_project [file join $project_dir u02_stream.xpr]
    if {[get_property PART [current_project]] ne "xczu5ev-sfvc784-2-i"} {error "Unexpected candidate part"}
    if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis phase incomplete"}
    if {$u02_partial_phase eq "implementation"} {
        if {[get_property STATUS [get_runs impl_1]] ne "Not started"} {error "Preserve existing implementation attempt"}
        set u02_partial_text [string range $u02_partial_text $u02_partial_impl_pos [expr {$u02_partial_report_pos - 1}]]
        append u02_partial_text "\nclose_project\nexit\n"
    } else {
        if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation phase incomplete"}
        set u02_partial_text [string range $u02_partial_text $u02_partial_report_pos end]
        append u02_partial_text "\nexit\n"
    }
    uplevel #0 $u02_partial_text
} else {error "Unsupported build phase"}
