# FPGA configuration and VIO observations only: no ARM MMIO, PS init or reset.
set root [file dirname [file normalize [info script]]]
cd $root
set bit [file join $root debug_project/u02_compute.runs/impl_1/compute_wrapper.bit]
set ltx [file join $root ../reports/clock_debug/compute.ltx]
if {![file exists $bit] || ![file exists $ltx]} {error "Validated debug artifacts missing"}
open_hw
connect_hw_server
set cable [get_hw_targets -filter {NAME =~ "*210512180081*"}]
if {[llength $cable] != 1} {error "Verified JTAG cable missing"}
current_hw_target [lindex $cable 0]
open_hw_target
set devices [get_hw_devices -filter {PART == xczu5}]
if {[llength $devices] != 1} {error "Expected ZU5 device"}
set device [lindex $devices 0]
set_property PROGRAM.FILE $bit $device
set_property PROBES.FILE $ltx $device
program_hw_devices $device
refresh_hw_device $device
if {[get_property REGISTER.CONFIG_STATUS.BIT13_DONE_INTERNAL_SIGNAL_STATUS $device] ne "1" ||
    [get_property REGISTER.CONFIG_STATUS.BIT00_CRC_ERROR $device] ne "0"} {error "Configuration failed"}
set vio [get_hw_vios -of_objects $device]
if {[llength $vio] != 1} {error "Independent-clock VIO not found"}
foreach probe [get_hw_probes -of_objects $vio] {report_property $probe}
for {set i 0} {$i < 3} {incr i} {
    refresh_hw_vio $vio
    foreach probe [get_hw_probes -of_objects $vio] {
        puts "U02_VIO_SAMPLE $i [get_property NAME $probe] [get_property INPUT_VALUE $probe]"
    }
    after 500
}
close_hw_target
disconnect_hw_server
close_hw
puts "U02_CLOCK_PROBE_OBSERVATION_DONE"
