set project [file normalize $::env(U02_DW128_PROJECT)]
set csim_log [file join $project zu5ev csim report u02_stream_128write_csim.log]
if {![file exists $csim_log]} {
    error "Complete same-project C simulation must precede synthesis"
}
set fp [open $csim_log r]
set evidence [read $fp]
close $fp
foreach marker {
    {U02_STREAM_ALL_REAL_GROUPS_PASSED groups=110 values=871872}
    {U02_DW128_RESULT_ALIGNMENT_PASSED cases=3}
    {CSim done with 0 errors.}
} {
    if {[string first $marker $evidence] < 0} {
        error "Missing required C simulation evidence: $marker"
    }
}
cd [file dirname $project]
open_project [file tail $project]
open_solution zu5ev
csynth_design
exit
