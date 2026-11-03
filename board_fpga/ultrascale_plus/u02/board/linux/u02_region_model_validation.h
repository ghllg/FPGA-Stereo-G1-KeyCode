#ifndef U02_REGION_MODEL_VALIDATION_H
#define U02_REGION_MODEL_VALIDATION_H
/* Validate a private, stable model snapshot. No floating-point operations,
 * userspace pointers, MMIO or DMA operations. Suitable for kernel inclusion. */
#ifdef __KERNEL__
#include <linux/types.h>
typedef u32 u02r_u32;
typedef s32 u02r_s32;
typedef u64 u02r_u64;
typedef s64 u02r_s64;
#else
#include <stdint.h>
#include <stddef.h>
typedef uint32_t u02r_u32;
typedef int32_t u02r_s32;
typedef uint64_t u02r_u64;
typedef int64_t u02r_s64;
#endif
static inline u02r_u32 u02r_word(const unsigned char *data, size_t at)
{
    return (u02r_u32)data[at] | ((u02r_u32)data[at+1]<<8) |
        ((u02r_u32)data[at+2]<<16) | ((u02r_u32)data[at+3]<<24);
}
static inline int u02r_scale_valid(u02r_u32 bits)
{
    return bits >= 0x2b800000u && bits <= 0x53800000u;
}
static inline int u02r_bias_valid(u02r_u32 bits)
{
    u02r_u32 magnitude = bits & 0x7fffffffu;
    return !magnitude || (magnitude >= 0x17800000u && magnitude <= 0x67800000u);
}
static inline int u02_region_model_validate(const void *raw, size_t bytes)
{
    const unsigned char *data = (const unsigned char *)raw;
    unsigned char seen[110] = {0};
    u02r_u32 count, i, j, k;
    u02r_u64 cursor;
    if (!data || bytes < 128 || bytes > 1048576 || bytes % 64) return -1;
    count = u02r_word(data,12);
    if (u02r_word(data,0)!=0x4d323055u || u02r_word(data,4)!=1 ||
        u02r_word(data,8)!=bytes || !count || count>16 || u02r_word(data,16)!=64 ||
        64+(u02r_u64)count*64>bytes) return -2;
    for (i=0;i<11;++i) if (u02r_word(data,20+i*4)) return -3;
    cursor=64+(u02r_u64)count*64;
    for (i=0;i<count;++i) {
        u02r_u32 g[16], p[8], offsets[3];
        u02r_s32 previous[12]={0};
        u02r_u64 weight_end=0;
        u02r_s32 expected_q=0;
        for (j=0;j<16;++j) g[j]=u02r_word(data,64+i*64+j*4);
        if (g[0]>=110 || seen[g[0]] || !g[6] || g[6]>320 || g[1]!=cursor ||
            g[1]%64 || g[2]<64 || g[2]>16384 || g[2]%64 || cursor+g[2]>bytes) return -4;
        seen[g[0]]=1;
        for (j=7;j<16;++j) if (g[j]) return -4;
        for (j=0;j<8;++j) p[j]=u02r_word(data,g[1]+j*4);
        if (p[0]!=0x55303253u || p[1]!=2 || !p[2] || p[2]>6 || p[5]!=g[2] ||
            p[3]<32+p[2]*48 || p[3]%64 || p[4]%64 || p[4]<p[3] || p[4]>p[5]) return -5;
        for (j=0;j<p[2];++j) {
            u02r_s32 op[12]; unsigned c;
            for (k=0;k<12;++k) op[k]=(u02r_s32)u02r_word(data,g[1]+32+j*48+k*4);
            if ((op[0]!=0 && op[0]!=1) || op[1]<1 || op[1]>320 || op[2]<1 || op[2]>320 ||
                op[3]<1 || op[3]>64 || op[4]<1 || op[4]>48 || op[5]<1 || op[5]>64 || op[6]<1 || op[6]>48 ||
                (op[7]!=1 && op[7]!=2) || op[8]<0 || op[8]>1 || (op[9]!=0 && op[9]!=1) ||
                op[10]<0 || op[10]%64 || op[11]!=expected_q) return -6;
            if ((!op[0] && (op[7]!=1 || op[8])) || (op[0] && op[1]!=op[2]) ||
                op[5]!=(op[3]+2*op[8]-(op[0]?3:1))/op[7]+1 ||
                op[6]!=(op[4]+2*op[8]-(op[0]?3:1))/op[7]+1 ||
                op[1]*op[3]*op[4]>98304 || op[2]*op[5]*op[6]>98304 ||
                op[5]*op[6]%2 || ((op[1]+15)/16)*((op[3]*op[4]+7)/8)>768 ||
                ((op[2]+15)/16)*((op[5]*op[6]+7)/8)>768) return -6;
            if ((u02r_u64)op[10]<weight_end || (u02r_u64)p[4]+op[10]+(u02r_u64)op[2]*(op[0]?9:op[1])>p[5] ||
                (u02r_u64)p[3]+((u02r_u64)op[11]+op[2])*16>p[4]) return -6;
            if (j && (previous[2]!=op[1] || previous[5]!=op[3] || previous[6]!=op[4])) return -6;
            if (!j && (g[6]!=(u02r_u32)op[2] || p[6]!=(u02r_u32)(op[1]*op[3]*op[4]))) return -6;
            weight_end=(u02r_u64)op[10]+(u02r_u64)op[2]*(op[0]?9:op[1]);
            for (c=0;c<(unsigned)op[2];++c) {
                size_t at=g[1]+p[3]+((size_t)op[11]+c)*16;
                u02r_u32 a=u02r_word(data,at), b=u02r_word(data,at+4), d=u02r_word(data,at+8), e=u02r_word(data,at+12);
                if (j+1<p[2]) {
                    u02r_s64 bias=(u02r_s64)((u02r_u64)d|((u02r_u64)e<<32));
                    if (a>=8388608 || b>40 || bias<-((u02r_s64)1<<60) || bias>((u02r_s64)1<<60)) return -7;
                } else if (!u02r_scale_valid(a) || !u02r_scale_valid(b) || !u02r_bias_valid(d) || e) return -7;
            }
            expected_q+=op[2];for (k=0;k<12;++k) previous[k]=op[k];
        }
        if (p[7]!=(u02r_u32)(previous[2]*previous[5]*previous[6])) return -6;
        cursor+=g[2];offsets[0]=g[3];offsets[1]=g[4];offsets[2]=g[5];
        for (j=0;j<3;++j) {
            if (offsets[j]!=cursor || cursor%64 || cursor+(u02r_u64)g[6]*4>bytes) return -8;
            for (k=0;k<g[6];++k) {
                u02r_u32 bits=u02r_word(data,(size_t)cursor+k*4);
                if (j==1 ? !u02r_bias_valid(bits) : !u02r_scale_valid(bits)) return -9;
            }
            cursor=(cursor+(u02r_u64)g[6]*4+63)/64*64;
        }
    }
    return cursor==bytes?0:-10;
}
#endif
