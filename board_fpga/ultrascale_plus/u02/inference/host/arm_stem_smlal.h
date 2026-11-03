#pragma once
#include <cstdint>
#if defined(__aarch64__)
#include <arm_neon.h>

// Exact signed INT8 products accumulated directly in INT32. Even the full
// INT8 range is bounded by 27*128*128=442368, so no partial sum overflows.
inline void u02_arm_stem_smlal(const int8_t* a,const int8_t* b,int32_t* c,int threads) {
    constexpr int m=32,n=3072,k=27;
    #pragma omp parallel for num_threads(threads)
    for(int row=0;row<m;row+=4)for(int col=0;col<n;col+=8) {
        int32x4_t l0=vdupq_n_s32(0),h0=l0,l1=l0,h1=l0,l2=l0,h2=l0,l3=l0,h3=l0;
        for(int inner=0;inner<k;++inner) {
            const int16x8_t pixels=vmovl_s8(vld1_s8(b+inner*n+col));
            const int16x4_t low=vget_low_s16(pixels);
            const int16_t w0=a[row*k+inner],w1=a[(row+1)*k+inner];
            const int16_t w2=a[(row+2)*k+inner],w3=a[(row+3)*k+inner];
            l0=vmlal_n_s16(l0,low,w0);h0=vmlal_high_n_s16(h0,pixels,w0);
            l1=vmlal_n_s16(l1,low,w1);h1=vmlal_high_n_s16(h1,pixels,w1);
            l2=vmlal_n_s16(l2,low,w2);h2=vmlal_high_n_s16(h2,pixels,w2);
            l3=vmlal_n_s16(l3,low,w3);h3=vmlal_high_n_s16(h3,pixels,w3);
        }
        vst1q_s32(c+row*n+col,l0);vst1q_s32(c+row*n+col+4,h0);
        vst1q_s32(c+(row+1)*n+col,l1);vst1q_s32(c+(row+1)*n+col+4,h1);
        vst1q_s32(c+(row+2)*n+col,l2);vst1q_s32(c+(row+2)*n+col+4,h2);
        vst1q_s32(c+(row+3)*n+col,l3);vst1q_s32(c+(row+3)*n+col+4,h3);
    }
}
#endif
