# Verify the corrected reset and AXI slave through JTAG before any ARM MMIO.
set root [file dirname [file normalize [info script]]]
cd $root
open_hw
connect_hw_server
set cable [get_hw_targets -filter {NAME =~ "*210512180081*"}]
if {[llength $cable] != 1} {error "Verified JTAG cable missing"}
current_hw_target [lindex $cable 0]
open_hw_target
set devices [get_hw_devices -filter {PART == xczu5}]
if {[llength $devices] != 1} {error "Expected ZU5 target"}
set device [lindex $devices 0]
set_property PROGRAM.FILE [file join $root debug_project/u02_compute.runs/impl_1/compute_wrapper.bit] $device
set_property PROBES.FILE [file join $root ../reports/reset_fixed/compute.ltx] $device
program_hw_devices $device
refresh_hw_device $device
if {[get_property REGISTER.CONFIG_STATUS.BIT13_DONE_INTERNAL_SIGNAL_STATUS $device] ne "1" ||
    [get_property REGISTER.CONFIG_STATUS.BIT00_CRC_ERROR $device] ne "0"} {error "Configuration status failed"}
set vio [get_hw_vios -of_objects $device]
if {[llength $vio] != 1} {error "Independent-clock VIO missing"}
for {set i 0} {$i < 3} {incr i} {
    refresh_hw_vio $vio
    foreach probe [get_hw_probes -of_objects $vio] {
        set name [get_property NAME $probe]
        set value [get_property INPUT_VALUE $probe]
        puts "U02_VIO_SAMPLE $i $name $value"
        if {[string match *clock_probe_resets $name] && $value ne "3"} {error "Fabric reset is still asserted"}
    }
    after 500
}
set axi [get_hw_axis -of_objects $device]
if {[llength $axi] != 1} {error "Independent JTAG AXI master missing"}
create_hw_axi_txn u02_read_ctrl [lindex $axi 0] -type READ -address A0000000 -len 1
run_hw_axi [get_hw_axi_txns u02_read_ctrl]
set control [get_property DATA [get_hw_axi_txns u02_read_ctrl]]
puts "U02_JTAG_AXI_CONTROL $control"
if {[expr 0x$control & 4] != 4} {error "HLS core is not idle"}
create_hw_axi_txn u02_write_count [lindex $axi 0] -type WRITE -address A0000030 -len 1 -data 00000007
run_hw_axi [get_hw_axi_txns u02_write_count]
create_hw_axi_txn u02_read_count [lindex $axi 0] -type READ -address A0000030 -len 1
run_hw_axi [get_hw_axi_txns u02_read_count]
set count [get_property DATA [get_hw_axi_txns u02_read_count]]
puts "U02_JTAG_AXI_COUNT_READBACK $count"
if {[expr 0x$count] != 7} {error "Independent AXI register readback failed"}
set_property DATA 00000000 [get_hw_axi_txns u02_write_count]
run_hw_axi [get_hw_axi_txns u02_write_count]
close_hw_target
disconnect_hw_server
close_hw
puts "U02_RESET_AND_AXI_INDEPENDENT_VERIFICATION_PASSED"
