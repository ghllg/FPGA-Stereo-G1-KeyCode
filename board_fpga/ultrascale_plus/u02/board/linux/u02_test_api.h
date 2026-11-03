#ifndef U02_TEST_API_H
#define U02_TEST_API_H
#include <linux/types.h>
#include <linux/ioctl.h>
struct u02_test {
    __u32 count, multiplier, addend, seed;
    __u32 timeout_us, status, errors, first_output;
    __aligned_u64 elapsed_ns;
    __aligned_u64 dma_address;
};
#define U02_RUN _IOWR('U', 2, struct u02_test)
#define U02_RECOVER _IO('U', 3)
#endif
