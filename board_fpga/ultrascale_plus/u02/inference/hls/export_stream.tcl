set root [file dirname [file normalize [info script]]]
cd [file join $root ..]
set project [file normalize [file join $root ../stream_hls_03]]
if {[info exists ::env(U02_STREAM_HLS_PROJECT)]} {set project [file normalize $::env(U02_STREAM_HLS_PROJECT)]}
cd [file dirname $project]
open_project [file tail $project]
open_solution zu5ev
if {![file exists [file join $project zu5ev syn report u02_stream_csynth.rpt]]} {error "Synthesis report missing"}
export_design -format ip_catalog -vendor u02 -library inference -version 1.0
exit
