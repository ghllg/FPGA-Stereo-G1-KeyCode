set project [file normalize $::env(U02_STREAM_HLS_PROJECT)]
if {![file exists [file join $project zu5ev csim report u02_stream_csim.log]]} {error "Missing C simulation evidence"}
set f [open [file join $project zu5ev csim report u02_stream_csim.log] r]
set proof [read $f];close $f
if {[string first "U02_STREAM_ALL_REAL_GROUPS_PASSED groups=110" $proof]<0} {error "Real whole-graph fixture gate did not pass"}
cd [file dirname $project]
open_project [file tail $project]
open_solution zu5ev
csynth_design
exit
