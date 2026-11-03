// Isolated diagnostic module. Keep ABI, DMA directions, ranges, guards,
// polling and timeout ownership identical to the formal resident driver.
#include <linux/module.h>
#include <linux/dma-mapping.h>
#include <linux/ktime.h>
#include "u02_stream_resident_api.h"

static unsigned long long profile_device_ns[3], profile_cpu_ns[3];
static unsigned long long profile_device_bytes[3], profile_cpu_bytes[3];
static unsigned long long profile_device_calls[3], profile_cpu_calls[3];
module_param_array_named(profile_device_ns, profile_device_ns, ullong, NULL, 0444);
module_param_array_named(profile_cpu_ns, profile_cpu_ns, ullong, NULL, 0444);
module_param_array_named(profile_device_bytes, profile_device_bytes, ullong, NULL, 0444);
module_param_array_named(profile_cpu_bytes, profile_cpu_bytes, ullong, NULL, 0444);
module_param_array_named(profile_device_calls, profile_device_calls, ullong, NULL, 0444);
module_param_array_named(profile_cpu_calls, profile_cpu_calls, ullong, NULL, 0444);

static unsigned int profile_region(unsigned long offset)
{
    return offset == U02_STREAM_RESIDENT_B_OFFSET ? 1 :
           offset == U02_STREAM_RESIDENT_C_OFFSET ? 2 : 0;
}

static void profile_sync_device(struct device *dev, dma_addr_t dma,
        unsigned long offset, size_t size, enum dma_data_direction direction)
{
    unsigned int region = profile_region(offset);
    u64 start = ktime_get_ns();
    dma_sync_single_range_for_device(dev, dma, offset, size, direction);
    profile_device_ns[region] += ktime_get_ns() - start;
    profile_device_bytes[region] += size;
    ++profile_device_calls[region];
}

static void profile_sync_cpu(struct device *dev, dma_addr_t dma,
        unsigned long offset, size_t size, enum dma_data_direction direction)
{
    unsigned int region = profile_region(offset);
    u64 start = ktime_get_ns();
    dma_sync_single_range_for_cpu(dev, dma, offset, size, direction);
    profile_cpu_ns[region] += ktime_get_ns() - start;
    profile_cpu_bytes[region] += size;
    ++profile_cpu_calls[region];
}

// Headers have already declared the real functions used by the wrappers.
// Only the formal source's explicit range calls are intercepted.
#define dma_sync_single_range_for_device profile_sync_device
#define dma_sync_single_range_for_cpu profile_sync_cpu
#include "u02_stream_resident.c"
