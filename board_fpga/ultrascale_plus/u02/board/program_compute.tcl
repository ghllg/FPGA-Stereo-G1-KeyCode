# Runtime PL programming only. No PS initialization, CPU reset, or boot-media write.
set root [file dirname [file normalize [info script]]]
set bit [file join $root compute_project/u02_compute.runs/impl_1/compute_wrapper.bit]
if {![file exists $bit]} {error "Compute bitstream missing"}
open_hw
connect_hw_server
set cables [get_hw_targets -filter {NAME =~ "*210512180081*"}]
if {[llength $cables] != 1} {error "Expected verified Digilent cable"}
current_hw_target [lindex $cables 0]
open_hw_target
set devices [get_hw_devices -filter {PART == xczu5}]
if {[llength $devices] != 1} {error "Expected one verified ZU5 target"}
set device [lindex $devices 0]
set_property PROGRAM.FILE $bit $device
program_hw_devices $device
refresh_hw_device $device
set done [get_property REGISTER.CONFIG_STATUS.BIT13_DONE_INTERNAL_SIGNAL_STATUS $device]
set crc [get_property REGISTER.CONFIG_STATUS.BIT00_CRC_ERROR $device]
puts "U02_PL_PROGRAM_STATUS DONE=$done CRC_ERROR=$crc"
if {$done ne "1" || $crc ne "0"} {error "PL configuration status failed"}
close_hw_target
disconnect_hw_server
close_hw
puts "U02_PL_PROGRAMMED"
