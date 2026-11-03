// Bounded restoration of the exact saved vendor RPLL state. No SMC or PL MMIO.
#include <linux/module.h>
#include <linux/io.h>
#include <linux/ioport.h>
#include <linux/iopoll.h>

#define RPLL_BASE 0xff5e0030ULL
#define RPLL_SIZE 0x14
static void __iomem *registers;
static bool apply;
module_param(apply, bool, 0444);
MODULE_PARM_DESC(apply, "Restore only the verified vendor RPLL from reset/bypass; do not change frequency");

static int __init restore_init(void)
{
    u32 saved, config, status;
    int result;
    if (!request_mem_region(RPLL_BASE, RPLL_SIZE, "u02_rpll_restore")) return -EBUSY;
    registers = ioremap(RPLL_BASE, RPLL_SIZE);
    if (!registers) { result = -ENOMEM; goto region; }
    saved = readl(registers);
    config = readl(registers + 4);
    status = readl(registers + 0x10);
    pr_info("u02_rpll_restore: control=%08x config=%08x PLL_STATUS=%08x apply=%u\n", saved, config, status, apply);
    if (!apply) return 0;
    // Exact psu_init.c baseline: ref=33.333MHz, feedback=64, divide-by-two.
    // Restore reset/bypass only; other PLLs, muxes and dividers are untouched.
    if (saved != 0x00014009 || (config & 0xfe7fedef) != (0x7e4b0c62 & 0xfe7fedef) || (status & BIT(1))) {
        result = -EINVAL; goto mapping;
    }
    writel(saved & ~BIT(0), registers);
    if (readl(registers) != (saved & ~BIT(0))) { result = -EACCES; goto rollback; }
    result = readl_poll_timeout(registers + 0x10, status, status & BIT(1), 10, 500000);
    if (result) goto rollback;
    writel(saved & ~(BIT(0) | BIT(3)), registers);
    if (readl(registers) != (saved & ~(BIT(0) | BIT(3)))) { result = -EACCES; goto rollback; }
    pr_info("U02_RPLL_VENDOR_STATE_RESTORED: control=%08x PLL_STATUS=%08x\n", readl(registers), readl(registers + 0x10));
    return 0;
rollback:
    // Enter bypass before reasserting reset, per the vendor sequence.
    writel(readl(registers) | BIT(3), registers);
    writel(saved, registers);
mapping:
    iounmap(registers);
region:
    release_mem_region(RPLL_BASE, RPLL_SIZE);
    return result;
}

static void __exit restore_exit(void)
{
    // Leave the restored clock running. Never remove a clock under active PL.
    iounmap(registers);
    release_mem_region(RPLL_BASE, RPLL_SIZE);
}
module_init(restore_init);
module_exit(restore_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Restore only a verified unused vendor RPLL, without firmware queries");
