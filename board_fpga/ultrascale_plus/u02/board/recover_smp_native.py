"""Retired diagnostic: persistent XSDB CPU polling produced a CPU0 APB fault.

Evidence is retained in reports/smp_native_recovery_1791745280492553300.
Do not execute this monitor again. Use normal Linux boot and a PL-only server,
and require actual four-core online snapshots before accepting measurements.
"""
if __name__ == '__main__':
    raise SystemExit('Retired CPU-polling recovery; inspect the preserved failure record.')
