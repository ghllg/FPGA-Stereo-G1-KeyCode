#pragma once
#include "quantize.h"
#if defined(U02_STREAM_GUARDED_QUANT)
#include "stream_guarded_quantize.h"
#endif

// AArch64 provides exact vector division and ties-away conversion. This
// removes the reciprocal-boundary detection and scalar fallback per vector.
inline void stream_quantize_array(const float* src,int8_t* dst,int count,float scale) {
#if defined(__aarch64__) && defined(U02_STREAM_GUARDED_QUANT)
    if(stream_guarded_quantize_array(src,dst,count,scale)) return;
#endif
#if defined(__aarch64__)
    if(scale>=1e-30f && scale<=1e30f) {
        const float32x4_t divisor=vdupq_n_f32(scale),lo=vdupq_n_f32(-127.f),hi=vdupq_n_f32(127.f);
        int i=0;
        for(;i+8<=count;i+=8) {
            const float32x4_t a=vmaxq_f32(lo,vminq_f32(hi,vdivq_f32(vld1q_f32(src+i),divisor)));
            const float32x4_t b=vmaxq_f32(lo,vminq_f32(hi,vdivq_f32(vld1q_f32(src+i+4),divisor)));
            const int16x8_t shorts=vcombine_s16(vmovn_s32(vcvtaq_s32_f32(a)),vmovn_s32(vcvtaq_s32_f32(b)));
            vst1_s8(dst+i,vmovn_s16(shorts));
        }
        for(;i<count;++i)dst[i]=pose_quantize(src[i],scale);
        return;
    }
#endif
    pose_quantize_array(src,dst,count,scale);
}
