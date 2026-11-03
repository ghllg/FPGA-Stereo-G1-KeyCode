# Reuse the verified C project; do not repeat C simulation or reset a solution.
set project [file normalize $::env(U02_FP32_HLS_PROJECT)]
if {![file exists $project]} {error "Verified C project missing"}
cd [file dirname $project]
open_project [file tail $project]
open_solution zu5ev
csynth_design
exit
