/* Isolated ABI9 IRQ completion. Included after the original driver's globals.
 * No boot/DT modification; no polling fallback; failure preserves inflight DMA.
 */
#include <linux/interrupt.h>
#include <linux/irq.h>
#include <linux/irqdomain.h>
#include <linux/of.h>
#include <linux/wait.h>
#include <linux/sysfs.h>
#include <linux/string.h>

static bool irq_route_checked;
module_param(irq_route_checked, bool, 0444);
MODULE_PARM_DESC(irq_route_checked, "Require actual DW128 routed PLPSIRQ0[0] and live GIC proof");
static unsigned int u02_completion_irq;
static bool u02_completion_mapping_owned, u02_completion_irq_requested;
static atomic_t u02_completion_masked = ATOMIC_INIT(1);
static DECLARE_WAIT_QUEUE_HEAD(u02_completion_wait);
static bool u02_completion_armed, u02_completion_done, u02_completion_error, u02_completion_failed;
static atomic64_t u02_irq_submitted = ATOMIC64_INIT(0), u02_irq_delivered = ATOMIC64_INIT(0);
static atomic64_t u02_irq_completed = ATOMIC64_INIT(0), u02_irq_unexpected = ATOMIC64_INIT(0);
static atomic64_t u02_irq_timeouts = ATOMIC64_INIT(0), u02_irq_recoveries = ATOMIC64_INIT(0);

/* Exactly one caller increments the disable depth. synchronize_irq in process
 * context then waits for a handler which may have won this atomic transition.
 * Every launch enables once, returning the descriptor to disabled depth one.
 */
static void u02_irq_mask_nosync(void)
{
    if (u02_completion_irq_requested && atomic_cmpxchg(&u02_completion_masked, 0, 1) == 0)
        disable_irq_nosync(u02_completion_irq);
}
static void u02_irq_mask_sync(void)
{
    u02_irq_mask_nosync();
    if (u02_completion_irq_requested) synchronize_irq(u02_completion_irq);
}
static void u02_irq_mask_hardware(void)
{
    writel(0, regs + 0x04);
    writel(0, regs + 0x08);
    (void)readl(regs + 0x08);
}
static int u02_irq_clear_idle(void)
{
    u32 ctrl = readl(regs), status;
    if ((ctrl & 1) || !(ctrl & 4)) return -EBUSY;
    status = readl(regs + 0x0c) & 3;
    if (status) writel(status, regs + 0x0c); /* TOW: never blindly write one. */
    return (readl(regs + 0x0c) & 3) ? -EIO : 0;
}
static irqreturn_t u02_completion_handler(int irq, void *opaque)
{
    u32 status, ctrl;
    bool armed;
    (void)irq; (void)opaque;
    /* Stop even an unexpected level source before examining AXI status. */
    u02_irq_mask_nosync();
    u02_irq_mask_hardware();
    status = readl(regs + 0x0c) & 3;
    ctrl = readl(regs); /* Clear AP_DONE latch, independently of ISR TOW. */
    if (status) writel(status, regs + 0x0c);
    (void)readl(regs + 0x0c);
    atomic64_inc(&u02_irq_delivered);
    armed = READ_ONCE(u02_completion_armed);
    if (armed && status == 1 && (ctrl & 2) && (ctrl & 4) && !(ctrl & 1)) {
        WRITE_ONCE(u02_completion_done, true);
        atomic64_inc(&u02_irq_completed);
    } else {
        WRITE_ONCE(u02_completion_error, true);
        WRITE_ONCE(u02_completion_failed, true);
        atomic64_inc(&u02_irq_unexpected);
    }
    WRITE_ONCE(u02_completion_armed, false);
    if (armed) wake_up(&u02_completion_wait);
    return IRQ_HANDLED;
}
static int u02_irq_prepare(void)
{
    int rc;
    if (READ_ONCE(u02_completion_failed)) return -EIO;
    u02_irq_mask_sync();
    u02_irq_mask_hardware();
    rc = u02_irq_clear_idle();
    if (rc) return rc;
    rc = irq_set_irqchip_state(u02_completion_irq, IRQCHIP_STATE_PENDING, false);
    if (rc) return rc; /* This exclusively owned GIC mapping must support it. */
    WRITE_ONCE(u02_completion_done, false);
    WRITE_ONCE(u02_completion_error, false);
    WRITE_ONCE(u02_completion_armed, false);
    return 0;
}
static int u02_irq_start_wait(u64 deadline)
{
    s64 remaining;
    int result, rc = 0;
    WRITE_ONCE(u02_completion_armed, true);
    writel(1, regs + 0x08); /* Done only; AP_READY must not complete a job. */
    writel(1, regs + 0x04);
    if (readl(regs + 0x08) != 1 || readl(regs + 0x04) != 1) {
        rc = -EIO;
        goto disarm;
    }
    atomic_set(&u02_completion_masked, 0);
    enable_irq(u02_completion_irq);
    atomic64_inc(&u02_irq_submitted);
    writel(1, regs);
    remaining = (s64)deadline - (s64)ktime_get_ns();
    if (remaining <= 0) result = -ETIME;
    else result = wait_event_hrtimeout(u02_completion_wait,
            READ_ONCE(u02_completion_done) || READ_ONCE(u02_completion_error), ns_to_ktime(remaining));
    if (result) { rc = -ETIMEDOUT; atomic64_inc(&u02_irq_timeouts); }
    else if (!READ_ONCE(u02_completion_done) || READ_ONCE(u02_completion_error)) rc = -EIO;
disarm:
    u02_irq_mask_sync();
    u02_irq_mask_hardware();
    WRITE_ONCE(u02_completion_armed, false);
    if (rc) WRITE_ONCE(u02_completion_failed, true);
    /* Caller never finish()/frees DMA on this error, including a late IRQ. */
    return rc;
}
static int u02_irq_recover(void)
{
    int rc;
    u02_irq_mask_sync(); u02_irq_mask_hardware();
    rc = u02_irq_clear_idle();
    if (!rc) atomic64_inc(&u02_irq_recoveries);
    /* Recovery releases DMA only; a failed candidate remains fail-closed. */
    return rc;
}
static void u02_irq_free(void)
{
    if (u02_completion_irq_requested) {
        u02_irq_mask_sync(); u02_irq_mask_hardware();
        free_irq(u02_completion_irq, &u02_completion_wait);
        u02_completion_irq_requested = false;
    }
    if (u02_completion_mapping_owned) {
        irq_dispose_mapping(u02_completion_irq);
        u02_completion_mapping_owned = false;
    }
}
static int u02_irq_init(void)
{
    struct device_node *gic;
    struct irq_domain *domain;
    struct irq_data *data;
    struct irq_fwspec spec = { .param_count = 3, .param = {0, 89, IRQ_TYPE_LEVEL_HIGH} };
    u32 cells;
    int rc;
    gic = of_find_node_by_path("/axi/interrupt-controller@f9010000");
    if (!gic) return -ENODEV;
    if (!of_device_is_compatible(gic, "arm,gic-400")) { of_node_put(gic); return -ENODEV; }
    if (!of_property_read_bool(gic, "interrupt-controller") ||
        of_property_read_u32(gic, "#interrupt-cells", &cells) || cells != 3) {
        of_node_put(gic); return -EINVAL;
    }
    domain = irq_find_host(gic);
    if (!domain) { of_node_put(gic); return -EPROBE_DEFER; }
    /* Never dispose or appropriate a mapping created by another driver. */
    if (irq_find_mapping(domain, 121)) { of_node_put(gic); return -EBUSY; }
    spec.fwnode = of_node_to_fwnode(gic);
    u02_completion_irq = irq_create_fwspec_mapping(&spec);
    of_node_put(gic);
    if (!u02_completion_irq) return -EINVAL;
    u02_completion_mapping_owned = true;
    data = irq_get_irq_data(u02_completion_irq);
    if (!data || data->domain != domain || data->hwirq != 121 ||
        irqd_get_trigger_type(data) != IRQ_TYPE_LEVEL_HIGH) { rc = -EINVAL; goto fail; }
    u02_irq_mask_hardware();
    rc = u02_irq_clear_idle();
    if (rc) goto fail;
    rc = request_irq(u02_completion_irq, u02_completion_handler, IRQF_TRIGGER_HIGH | IRQF_NO_AUTOEN,
                     "u02_stream_pipeline_irq", &u02_completion_wait);
    if (rc) goto fail;
    u02_completion_irq_requested = true;
    return 0;
fail:
    u02_irq_free();
    return rc;
}
static ssize_t irq_stats_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    (void)dev; (void)attr;
    return sysfs_emit(buf, "{\"linux_irq\":%u,\"gic_hwirq\":121,\"spi\":89,\"submitted\":%lld,\"delivered\":%lld,\"completed\":%lld,\"unexpected\":%lld,\"timeouts\":%lld,\"recoveries\":%lld,\"failed\":%u}\n",
        u02_completion_irq, (long long)atomic64_read(&u02_irq_submitted), (long long)atomic64_read(&u02_irq_delivered),
        (long long)atomic64_read(&u02_irq_completed), (long long)atomic64_read(&u02_irq_unexpected),
        (long long)atomic64_read(&u02_irq_timeouts), (long long)atomic64_read(&u02_irq_recoveries),
        READ_ONCE(u02_completion_failed) ? 1 : 0);
}
static DEVICE_ATTR_RO(irq_stats);
static struct attribute *u02_irq_attrs[] = { &dev_attr_irq_stats.attr, NULL };
ATTRIBUTE_GROUPS(u02_irq);
