# Reuse proven PS/clock/reset/address construction in a separate candidate.
if {![info exists ::env(U02_FP32_IP_PACKAGE)]} {error "Verified FP32 IP path required"}
set u02_fp32_base_script [file join [file dirname [file normalize [info script]]] create_stream_project.tcl]
set u02_fp32_channel [open $u02_fp32_base_script r]
set u02_fp32_text [read $u02_fp32_channel]
close $u02_fp32_channel
set u02_fp32_needle {[file join $hls_project zu5ev/impl/ip]}
if {[string first $u02_fp32_needle $u02_fp32_text] < 0} {
    error "Missing proven package-path binding"
}
set u02_fp32_text [string map [list $u02_fp32_needle {[file normalize $::env(U02_FP32_IP_PACKAGE)]}] $u02_fp32_text]
uplevel #0 $u02_fp32_text
