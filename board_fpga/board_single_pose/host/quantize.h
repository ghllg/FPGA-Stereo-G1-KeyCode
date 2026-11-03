#pragma once
#include <algorithm>
#include <stdint.h>
#include <cmath>
#include <limits>
#include <cstring>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Identical half-away-from-zero rounding on the clamped finite [-127,127]
// quotient, without a roundf library call for every activation.
inline int8_t pose_quantize(float value, float scale) {
    const float q=std::max(-127.f,std::min(127.f,value/scale));
    const int whole=static_cast<int>(q);
    const float fraction=q-static_cast<float>(whole);
    return int8_t(whole+int(fraction>=0.5f)-int(fraction<=-0.5f));
}

// A rounded reciprocal/multiply can differ from division near half-integers.
// Use scalar division for those lanes and for extreme scales. For normal
// bounded quotients, reciprocal+multiply error is under 4e-5; the 1e-3
// half-integer guard also covers division rounding. No fast-math is required.
inline void pose_quantize_array(const float* src,int8_t* dst,int count,float scale) {
    int j=0;
#ifdef __ARM_NEON
    if(scale>=1e-30f && scale<=1e30f) {
        const float reciprocal=1.f/scale;
        for(;j+4<=count;j+=4) {
            float32x4_t q=vmulq_n_f32(vld1q_f32(src+j),reciprocal);
            q=vmaxq_f32(vdupq_n_f32(-127.f),vminq_f32(vdupq_n_f32(127.f),q));
            int32x4_t whole=vcvtq_s32_f32(q);
            float32x4_t frac=vsubq_f32(q,vcvtq_f32_s32(whole));
            const uint32x4_t pos=vcgeq_f32(frac,vdupq_n_f32(0.5f));
            const uint32x4_t neg=vcleq_f32(frac,vdupq_n_f32(-0.5f));
            int32x4_t rounded=vsubq_s32(vaddq_s32(whole,vreinterpretq_s32_u32(neg)),vreinterpretq_s32_u32(pos));
            uint32x4_t near=vcltq_f32(vabsq_f32(vsubq_f32(vabsq_f32(frac),vdupq_n_f32(0.5f))),vdupq_n_f32(1e-3f));
            uint32x2_t any=vorr_u32(vget_low_u32(near),vget_high_u32(near));
            if(vget_lane_u32(any,0)|vget_lane_u32(any,1)) {
                for(int lane=0;lane<4;++lane) dst[j+lane]=pose_quantize(src[j+lane],scale);
            } else {
                int16x4_t narrow=vmovn_s32(rounded);
                int8x8_t bytes=vmovn_s16(vcombine_s16(narrow,vdup_n_s16(0)));
                int32_t packed=vget_lane_s32(vreinterpret_s32_s8(bytes),0);
                std::memcpy(dst+j,&packed,4);
            }
        }
    }
#endif
    for(;j<count;++j) dst[j]=pose_quantize(src[j],scale);
}

inline bool pose_max_abs(const float* src,int count,float& maximum) {
    int j=0;
#ifdef __ARM_NEON
    uint32_t old_bits;std::memcpy(&old_bits,&maximum,4);
    uint32x4_t vmax=vdupq_n_u32(old_bits), vmax2=vmax;
    const uint32x4_t abs_mask=vdupq_n_u32(0x7fffffffu);
    for(;j+8<=count;j+=8) {
        vmax=vmaxq_u32(vmax,vandq_u32(vreinterpretq_u32_f32(vld1q_f32(src+j)),abs_mask));
        vmax2=vmaxq_u32(vmax2,vandq_u32(vreinterpretq_u32_f32(vld1q_f32(src+j+4)),abs_mask));
    }
    vmax=vmaxq_u32(vmax,vmax2);
    for(;j+4<=count;j+=4) {
        vmax=vmaxq_u32(vmax,vandq_u32(vreinterpretq_u32_f32(vld1q_f32(src+j)),abs_mask));
    }
    uint32x2_t half=vmax_u32(vget_low_u32(vmax),vget_high_u32(vmax));
    uint32_t bits=std::max(vget_lane_u32(half,0),vget_lane_u32(half,1));
    // Positive IEEE-754 bit patterns sort numerically. Integer comparisons
    // preserve subnormals and identify both infinities and NaNs.
    if(bits>=0x7f800000u) return false;
    std::memcpy(&maximum,&bits,4);
#endif
    for(;j<count;++j) {
        if(!std::isfinite(src[j])) return false;
        maximum=std::max(maximum,std::abs(src[j]));
    }
    return true;
}
