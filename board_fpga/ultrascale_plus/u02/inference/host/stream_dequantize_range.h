#pragma once
#include "stream_dequantize.h"
#include "quantize.h"
#include <cstring>

// Same rounded output arithmetic as stream_dequantize_array, with the exact
// integer-bit absolute reduction used by pose_max_abs while values are live.
inline bool stream_dequantize_with_range(const int32_t* src,float* dst,int count,
        float incoming,float weight,float bias,bool relu,float& maximum) {
    int i=0;maximum=0;
#if defined(__aarch64__)
    const auto si=vdupq_n_f32(incoming),sw=vdupq_n_f32(weight),sb=vdupq_n_f32(bias),zero=vdupq_n_f32(0.f);
    const auto mask=vdupq_n_u32(0x7fffffffu);auto vmax=vdupq_n_u32(0);
    for(;i+4<=count;i+=4) {
        auto value=vcvtq_f32_s32(vld1q_s32(src+i));
        value=vmulq_f32(value,si);value=vmulq_f32(value,sw);value=vaddq_f32(value,sb);
        if(relu)value=vbslq_f32(vcgtq_f32(value,zero),value,zero);
        vst1q_f32(dst+i,value);
        vmax=vmaxq_u32(vmax,vandq_u32(vreinterpretq_u32_f32(value),mask));
    }
    uint32x2_t half=vmax_u32(vget_low_u32(vmax),vget_high_u32(vmax));
    const uint32_t bits=std::max(vget_lane_u32(half,0),vget_lane_u32(half,1));
    // Finish all output writes even when a range cannot be cached.
    if(bits<0x7f800000u)std::memcpy(&maximum,&bits,4);
#endif
    const int tail=i;
    for(;i<count;++i) {
        float value=float(src[i])*incoming;value=value*weight;value=value+bias;
        dst[i]=relu?std::max(0.f,value):value;
    }
#if defined(__aarch64__)
    if(bits>=0x7f800000u)return false;
#endif
    return pose_max_abs(dst+tail,count-tail,maximum);
}
