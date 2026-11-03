set project [file normalize $::env(U02_REGION_HLS_PROJECT)]
cd [file dirname $project]
open_project [file tail $project]
open_solution zu5ev
csim_design -O
# Generate the preserved five-transaction vectors and C postchecker only.
# XSIM elaboration/execution is a separate checked, single-thread stage.
cosim_design -rtl verilog -tool xsim -setup -trace_level none
puts "U02_REGION_DUAL_RTL_SETUP_FINISHED"
exit
