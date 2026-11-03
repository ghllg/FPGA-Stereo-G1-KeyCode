# Read-only DAP inspection of addresses/masks from the saved vendor psu_init.h/c.
# Never run psu_init, stop a core, reset PS, or issue mwr from this script.
connect
puts "U02_DEBUG_TARGETS [targets]"
targets -set -filter {name == "PSU"}
puts "U02_PL0_REF_CTRL [mrd -force 0xFF5E00C0 1]"
puts "U02_GPIO_RESET_DATA [mrd -force 0xFF0A0054 1]"
puts "U02_GPIO_RESET_DIRECTION [mrd -force 0xFF0A0344 1]"
puts "U02_GPIO_RESET_ENABLE [mrd -force 0xFF0A0348 1]"
puts "U02_AFI_RESET [mrd -force 0xFD1A0100 1]"
disconnect
