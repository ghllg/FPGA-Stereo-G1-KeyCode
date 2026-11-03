#pragma once
#include <algorithm>
#include <cstdint>
#ifdef U02_STREAM_DEQUANT_UNROLL4
#include "stream_dequantize_unrolled.h"
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// Preserve the reference's two rounded multiplies followed by an add.
// Compile callers with -ffp-contract=off: fused multiply-add changes results.
inline void stream_dequantize_array(const int32_t* src,float* dst,int count,
                                   float incoming,float weight,float bias,bool relu) {
#ifdef U02_STREAM_DEQUANT_UNROLL4
    stream_dequantize_unrolled(src,dst,count,incoming,weight,bias,relu);
#else
    int i=0;
#if defined(__aarch64__)
    const float32x4_t si=vdupq_n_f32(incoming),sw=vdupq_n_f32(weight),sb=vdupq_n_f32(bias);
    const float32x4_t zero=vdupq_n_f32(0.f);
    for(;i+4<=count;i+=4) {
        float32x4_t value=vcvtq_f32_s32(vld1q_s32(src+i));
        value=vmulq_f32(value,si);
        value=vmulq_f32(value,sw);
        value=vaddq_f32(value,sb);
        // This also returns +0 for -0 or NaN, matching std::max(0.f,value).
        if(relu)value=vbslq_f32(vcgtq_f32(value,zero),value,zero);
        vst1q_f32(dst+i,value);
    }
#endif
    for(;i<count;++i) {
        float value=float(src[i])*incoming;
        value=value*weight;value=value+bias;
        dst[i]=relu?std::max(0.f,value):value;
    }
#endif
}
