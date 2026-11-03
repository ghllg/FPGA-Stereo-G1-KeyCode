# Conservative upper clock bound for the internal configuration oscillator.
create_clock -name cfg_debug -period 15.0 [get_pins -hier -filter {REF_PIN_NAME == CFGMCLK}]
# Probe data has two receiving flops; it is diagnostic, never a DMA handshake.
set_clock_groups -asynchronous -group [get_clocks cfg_debug] -group [get_clocks clk_pl_0]
