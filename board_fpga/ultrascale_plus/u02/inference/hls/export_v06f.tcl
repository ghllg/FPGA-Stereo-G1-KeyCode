# Export only the already validated ZU5EV synthesis; do not resynthesize old sources.
set root [file dirname [file normalize [info script]]]
cd [file join $root ..]
open_project v06f_hls
open_solution zu5ev
export_design -format ip_catalog -vendor u02 -library inference -version 1.0
exit
