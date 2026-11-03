# Export a checked solution copy; never rerun synthesis or C/RTL simulation.
set project [file normalize $::env(U02_REGION_EXPORT_PROJECT)]
cd [file dirname $project]
open_project [file tail $project]
open_solution zu5ev
if {![file isfile [file join $project zu5ev syn report u02_region_dual_csynth.rpt]]} {
    error "Completed region synthesis required"
}
export_design -format ip_catalog -vendor u02 -library inference -version 1.0
puts U02_REGION_DUAL_EXPORT_FINISHED
exit
