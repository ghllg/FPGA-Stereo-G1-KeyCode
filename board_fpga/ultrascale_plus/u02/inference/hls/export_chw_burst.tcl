set project [file normalize $::env(U02_STREAM_HLS_PROJECT)]
if {![file exists [file join $project zu5ev syn report u02_stream_chw_burst_csynth.rpt]]} {
    error "Reviewed candidate synthesis report missing"
}
cd [file dirname $project]
open_project [file tail $project]
open_solution zu5ev
export_design -format ip_catalog -vendor u02 -library inference -version 1.0
exit
