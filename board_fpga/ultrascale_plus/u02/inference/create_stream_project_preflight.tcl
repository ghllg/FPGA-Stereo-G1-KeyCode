# Read-only tool-resource preflight after a reproduced PS customization failure.
set xgui {D:/Develop/Xilink/Vivado/2018.3/data/rsb/iprepos/intf_aximm_v1_0/xgui/aximm_v1_0.tcl}
if {![file isfile $xgui] || ![file readable $xgui]} {error "Installed AXI interface XGUI cannot be read: $xgui"}
set fd [open $xgui r]
set length [string length [read $fd]]
close $fd
if {$length < 1000} {error "Installed AXI interface XGUI is incomplete"}
puts "U02_STREAM_XGUI_READ_PREFLIGHT_PASSED bytes=$length"
source [file join [file dirname [info script]] create_stream_project.tcl]
