"""Inspect the running board through its existing UART without changing boot or PL."""
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bringup/axu5evb_e_boot"))
from uart_console import Console


def main():
    reports = Path(__file__).resolve().parents[1] / "reports"
    destination = reports / ("board_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    commands = {
        "identity": "uname -a; cat /proc/cmdline; cat /proc/device-tree/model",
        "memory": "cat /proc/iomem; cat /proc/meminfo",
        "runtime": "ls -l /sys/class/fpga_manager /sys/class/uio /dev/udmabuf* /lib/modules 2>&1; cat /proc/modules",
        "kernel": 'zcat /proc/config.gz | grep -E "CONFIG_(MODULES|MODVERSIONS|MODULE_SIG|CMA|DMA_CMA|DMA_SHARED_BUFFER|UIO|OF_OVERLAY|FPGA|ZYNQMP_FIRMWARE|ZYNQMP_PM_DOMAINS)"',
        "clocks": "ls -l /sys/kernel/debug/clk 2>&1; test ! -r /sys/kernel/debug/clk/clk_summary || cat /sys/kernel/debug/clk/clk_summary",
        "network": "ip addr show; cat /sys/class/net/eth0/carrier",
    }
    with (destination / "uart.log").open("wb") as log:
        console = Console("COM9", log)
        try:
            console.sync()
            result = {}
            for name, command in commands.items():
                status, output = console.run(command)
                result[name] = {"command": command, "exit_code": status, "output": output}
                (destination / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"PROBE {name} exit={status}", flush=True)
        finally:
            console.port.close()
    print(destination, flush=True)


if __name__ == "__main__":
    main()
