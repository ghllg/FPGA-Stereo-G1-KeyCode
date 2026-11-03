#pragma once
#include <algorithm>
#include <cstdint>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// Four independent vectors expose instruction-level parallelism on A53.
// Each lane still executes int->float, multiply, multiply, add in that order.
// No reassociation/FMA, changed scale, extra range scan or shared scratch.
inline void stream_dequantize_unrolled(const int32_t* src,float* dst,int count,
                                      float incoming,float weight,float bias,bool relu) {
    int i=0;
#if defined(__aarch64__)
    const float32x4_t si=vdupq_n_f32(incoming),sw=vdupq_n_f32(weight),sb=vdupq_n_f32(bias);
    const float32x4_t zero=vdupq_n_f32(0.f);
    for(;i+16<=count;i+=16) {
        float32x4_t a=vcvtq_f32_s32(vld1q_s32(src+i));
        float32x4_t b=vcvtq_f32_s32(vld1q_s32(src+i+4));
        float32x4_t c=vcvtq_f32_s32(vld1q_s32(src+i+8));
        float32x4_t d=vcvtq_f32_s32(vld1q_s32(src+i+12));
        a=vmulq_f32(a,si);b=vmulq_f32(b,si);c=vmulq_f32(c,si);d=vmulq_f32(d,si);
        a=vmulq_f32(a,sw);b=vmulq_f32(b,sw);c=vmulq_f32(c,sw);d=vmulq_f32(d,sw);
        a=vaddq_f32(a,sb);b=vaddq_f32(b,sb);c=vaddq_f32(c,sb);d=vaddq_f32(d,sb);
        if(relu) {
            a=vbslq_f32(vcgtq_f32(a,zero),a,zero);b=vbslq_f32(vcgtq_f32(b,zero),b,zero);
            c=vbslq_f32(vcgtq_f32(c,zero),c,zero);d=vbslq_f32(vcgtq_f32(d,zero),d,zero);
        }
        vst1q_f32(dst+i,a);vst1q_f32(dst+i+4,b);vst1q_f32(dst+i+8,c);vst1q_f32(dst+i+12,d);
    }
    for(;i+4<=count;i+=4) {
        float32x4_t value=vcvtq_f32_s32(vld1q_s32(src+i));
        value=vmulq_f32(value,si);value=vmulq_f32(value,sw);value=vaddq_f32(value,sb);
        if(relu)value=vbslq_f32(vcgtq_f32(value,zero),value,zero);
        vst1q_f32(dst+i,value);
    }
#endif
    for(;i<count;++i) {
        float value=float(src[i])*incoming;
        value=value*weight;value=value+bias;
        dst[i]=relu?std::max(0.f,value):value;
    }
}
