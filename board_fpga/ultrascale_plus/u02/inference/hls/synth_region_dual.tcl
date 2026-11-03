set project [file normalize $::env(U02_REGION_HLS_PROJECT)]
if {![file exists $project]} {error "Verified dual-mode C project missing"}
cd [file dirname $project]
open_project [file tail $project]
open_solution zu5ev
csynth_design
exit
