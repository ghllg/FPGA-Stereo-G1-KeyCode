#pragma once
#include "stream_quantize.h"
#if defined(U02_STREAM_BOUNDED_GUARDED_QUANT) && !defined(U02_STREAM_BOUNDED_QUANT)
#error Bounded reciprocal guard requires the finite maximum-based bounded path
#endif

// Contract: a successful finite input_maximum scan has proved
// abs(src[i]) <= maximum for every element. Keep the existing exact division
// and ties-away conversion. A conservative bound below 127.5 makes clipping
// redundant for the final integer, including directed FPCR rounding modes.
// This helper must not be used with unscanned or independently calibrated data.
inline void stream_quantize_bounded_array(const float* src,int8_t* dst,int count,
                                          float scale,float maximum) {
#if defined(__aarch64__) && defined(U02_STREAM_BOUNDED_QUANT)
    if(scale>=1e-30f && scale<=1e30f && maximum>=0.f &&
       maximum<=scale*127.25f) {
        const float32x4_t divisor=vdupq_n_f32(scale);
        int i=0;
#ifdef U02_STREAM_BOUNDED_GUARDED_QUANT
        // The successful finite/max scan above bounds |x/scale| <=127.25
        // (plus small IEEE rounding). Normal scale and reciprocal keep the
        // reciprocal-product versus division error below 5e-5 under all FPCR
        // rounding modes (a conservative three-rounding bound).
        // A 1e-3 distance guard uses exact division for an entire eight-lane
        // block whenever a ties-away byte could change. No scale calibration,
        // approximate byte acceptance at a boundary, or clipping is introduced.
        const float32x4_t reciprocal=vdupq_n_f32(1.f/scale),edge=vdupq_n_f32(.499f);
        for(;i+8<=count;i+=8) {
            const float32x4_t x=vld1q_f32(src+i),y=vld1q_f32(src+i+4);
            const float32x4_t a=vmulq_f32(x,reciprocal),b=vmulq_f32(y,reciprocal);
            int32x4_t ra=vcvtaq_s32_f32(a),rb=vcvtaq_s32_f32(b);
            const uint32x4_t near_a=vcgeq_f32(vabsq_f32(vsubq_f32(a,vcvtq_f32_s32(ra))),edge);
            const uint32x4_t near_b=vcgeq_f32(vabsq_f32(vsubq_f32(b,vcvtq_f32_s32(rb))),edge);
            if(vmaxvq_u32(vorrq_u32(near_a,near_b))) {
                ra=vcvtaq_s32_f32(vdivq_f32(x,divisor));
                rb=vcvtaq_s32_f32(vdivq_f32(y,divisor));
            }
            vst1_s8(dst+i,vmovn_s16(vcombine_s16(vmovn_s32(ra),vmovn_s32(rb))));
        }
#else
        for(;i+16<=count;i+=16) {
            const float32x4_t a=vdivq_f32(vld1q_f32(src+i),divisor);
            const float32x4_t b=vdivq_f32(vld1q_f32(src+i+4),divisor);
            const float32x4_t c=vdivq_f32(vld1q_f32(src+i+8),divisor);
            const float32x4_t d=vdivq_f32(vld1q_f32(src+i+12),divisor);
            const int16x8_t ab=vcombine_s16(vmovn_s32(vcvtaq_s32_f32(a)),vmovn_s32(vcvtaq_s32_f32(b)));
            const int16x8_t cd=vcombine_s16(vmovn_s32(vcvtaq_s32_f32(c)),vmovn_s32(vcvtaq_s32_f32(d)));
            vst1q_s8(dst+i,vcombine_s8(vmovn_s16(ab),vmovn_s16(cd)));
        }
        for(;i+8<=count;i+=8) {
            const float32x4_t a=vdivq_f32(vld1q_f32(src+i),divisor);
            const float32x4_t b=vdivq_f32(vld1q_f32(src+i+4),divisor);
            vst1_s8(dst+i,vmovn_s16(vcombine_s16(vmovn_s32(vcvtaq_s32_f32(a)),vmovn_s32(vcvtaq_s32_f32(b)))));
        }
#endif
#ifdef U02_STREAM_QUANT_TAIL4
        if(i+4<=count) {
            const float32x4_t q=vdivq_f32(vld1q_f32(src+i),divisor);
            const int16x4_t shorts=vmovn_s32(vcvtaq_s32_f32(q));
            const int8x8_t bytes=vmovn_s16(vcombine_s16(shorts,vdup_n_s16(0)));
            const int32_t packed=vget_lane_s32(vreinterpret_s32_s8(bytes),0);
            std::memcpy(dst+i,&packed,sizeof(packed));i+=4;
        }
#endif
        for(;i<count;++i)dst[i]=pose_quantize(src[i],scale);
        return;
    }
#endif
    stream_quantize_array(src,dst,count,scale);
}
