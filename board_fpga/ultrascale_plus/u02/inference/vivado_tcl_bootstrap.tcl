# Initialize only this Vivado process using its own bundled Tcl library.
# No installation files or global environment are changed.
set task_library [file normalize [info library]]
puts "U02_VIVADO_TCL_LIBRARY $task_library"
if {$task_library ne "D:/Develop/Xilink/Vivado/2018.3/tps/tcl/tcl8.5"} {
    error "Unexpected Tcl library; do not mix tool/runtime versions"
}
set task_tm [file join $task_library tm.tcl]
if {![file exists $task_tm]} {error "Bundled Tcl module loader missing"}
source $task_tm
if {![llength [info commands ::tcl::tm::UnknownHandler]]} {
    error "Bundled Tcl module handler did not initialize"
}
puts "U02_VIVADO_TCL_MSGCAT [package require msgcat]"
puts U02_VIVADO_TCL_BOOTSTRAP_PASSED
