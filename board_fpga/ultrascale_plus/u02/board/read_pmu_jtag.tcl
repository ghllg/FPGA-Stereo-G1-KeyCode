# Read-only attempt to inspect PMU isolation state without ARM firmware calls.
connect
puts "U02_PMU_TARGETS [targets]"
puts "U02_PMU_PROPERTIES [targets -filter {name =~ "*PMU*"} -target-properties]"
targets -set -filter {name =~ "*PMU*"}
puts "U02_PMU_PWR_STATE [mrd -force 0xFFD80100 1]"
puts "U02_PMU_AIB_STATE [mrd -force 0xFFD80604 1]"
disconnect
