# Reuse read-only vendor PS settings and explicitly enable the validated AXI paths.
# This file only configures a new offline BD; it must never run psu_init on Linux.
proc configure_u02_ps {cell repo} {
    source [file join $repo bringup/axu5evb_e_boot/vendor/ps_config.tcl]
    set_ps_config $cell
    set_property -dict [list \
        CONFIG.PSU__USE__M_AXI_GP0 {1} \
        CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
        CONFIG.PSU__USE__IRQ0 {1} \
        CONFIG.PSU__FPGA_PL0_ENABLE {1}] [get_bd_cells $cell]
}
