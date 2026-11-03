#pragma once
#include "quantize.h"

#if defined(__aarch64__)
// Finite activations are checked by pose_max_abs before this function.
// With normal scale/reciprocal and clamped |q| <= 127, reciprocal/multiply
// versus division rounding differs by less than 4e-5. A 1e-3 margin to every
// half-integer therefore uses exact division whenever a rounded byte might
// differ. This is a guarded arithmetic path, not a changed quantization scale.
inline bool stream_guarded_quantize_array(const float* src,int8_t* dst,int count,float scale) {
    if(!(scale>=1e-30f && scale<=1e30f)) return false;
    const float32x4_t reciprocal=vdupq_n_f32(1.f/scale), divisor=vdupq_n_f32(scale);
    const float32x4_t lo=vdupq_n_f32(-127.f),hi=vdupq_n_f32(127.f),edge=vdupq_n_f32(0.499f);
    int i=0;
    for(;i+8<=count;i+=8) {
        const float32x4_t x=vld1q_f32(src+i),y=vld1q_f32(src+i+4);
        const float32x4_t a=vmaxq_f32(lo,vminq_f32(hi,vmulq_f32(x,reciprocal)));
        const float32x4_t b=vmaxq_f32(lo,vminq_f32(hi,vmulq_f32(y,reciprocal)));
        int32x4_t ra=vcvtaq_s32_f32(a),rb=vcvtaq_s32_f32(b);
        const uint32x4_t near_a=vcgeq_f32(vabsq_f32(vsubq_f32(a,vcvtq_f32_s32(ra))),edge);
        const uint32x4_t near_b=vcgeq_f32(vabsq_f32(vsubq_f32(b,vcvtq_f32_s32(rb))),edge);
        if(vmaxvq_u32(vorrq_u32(near_a,near_b))) {
            ra=vcvtaq_s32_f32(vmaxq_f32(lo,vminq_f32(hi,vdivq_f32(x,divisor))));
            rb=vcvtaq_s32_f32(vmaxq_f32(lo,vminq_f32(hi,vdivq_f32(y,divisor))));
        }
        const int16x8_t shorts=vcombine_s16(vmovn_s32(ra),vmovn_s32(rb));
        vst1_s8(dst+i,vmovn_s16(shorts));
    }
    for(;i<count;++i) dst[i]=pose_quantize(src[i],scale);
    return true;
}
#endif
