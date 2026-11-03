add_files [file join $root clock_probe.v]
create_bd_cell -type module -reference u02_clock_probe clock_probe
create_bd_cell -type ip -vlnv xilinx.com:ip:vio:3.0 clock_vio
set_property -dict [list CONFIG.C_NUM_PROBE_IN {3} CONFIG.C_NUM_PROBE_OUT {0} CONFIG.C_PROBE_IN0_WIDTH {32} CONFIG.C_PROBE_IN1_WIDTH {32} CONFIG.C_PROBE_IN2_WIDTH {2}] [get_bd_cells clock_vio]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins clock_probe/ps_clk]
connect_bd_net [get_bd_pins por/reset_n] [get_bd_pins clock_probe/por_n]
connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins clock_probe/fabric_reset_n]
connect_bd_net [get_bd_pins clock_probe/debug_clk] [get_bd_pins clock_vio/clk]
connect_bd_net [get_bd_pins clock_probe/debug_count] [get_bd_pins clock_vio/probe_in0]
connect_bd_net [get_bd_pins clock_probe/pl_gray] [get_bd_pins clock_vio/probe_in1]
connect_bd_net [get_bd_pins clock_probe/resets] [get_bd_pins clock_vio/probe_in2]
add_files -fileset constrs_1 [file join $root clock_probe.xdc]
