#ifndef U02_REGION_LINUX_MEMORY_H
#define U02_REGION_LINUX_MEMORY_H
#ifndef __KERNEL__
#error Region DMA adapter is kernel-only
#endif
#include <linux/dma-mapping.h>
#include <linux/uaccess.h>
#include <linux/kernel.h>
#include "u02_region_session.h"

/* context is the actual device whose DMA mask and PS/PL access were verified.
 * Allocation is private: never expose these buffers through mmap. */
static int u02_region_dma_allocate(void *context,size_t bytes,struct u02_region_buffer *buffer)
{
    struct device *device=context;
    dma_addr_t address;
    buffer->cpu=dma_alloc_coherent(device,bytes,&address,GFP_KERNEL);
    if (!buffer->cpu) return -ENOMEM;
    buffer->dma=(u02r_u64)address;buffer->capacity=bytes;
    return 0;
}
static void u02_region_dma_release(void *context,struct u02_region_buffer *buffer)
{
    dma_free_coherent((struct device *)context,buffer->capacity,buffer->cpu,(dma_addr_t)buffer->dma);
    memset(buffer,0,sizeof(*buffer));
}
static int u02_region_copy_user(void *context,void *destination,u02r_u64 source,size_t bytes)
{
    (void)context;
    return copy_from_user(destination,u64_to_user_ptr(source),bytes) ? -EFAULT : 0;
}
static const struct u02_region_memory_ops u02_region_linux_memory_ops={
    .allocate=u02_region_dma_allocate,.release=u02_region_dma_release,.copy_user=u02_region_copy_user,
};
#endif
