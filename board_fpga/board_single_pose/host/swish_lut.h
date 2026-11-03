#pragma once
#include <array>
#include <cmath>
#include <limits>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Optional bounded approximation, not bit-exact expf/NEON Swish.
// [-20,20] at 1/128 spacing; interpolation error is verified separately.
struct PoseSwishLut {
    std::array<float,5121> values;
    PoseSwishLut() {
        for(int i=0;i<=5120;++i) {
            const double x=double(i)/128.0-20.0;
            values[i]=float(x/(1.0+std::exp(-x)));
        }
    }
    float operator()(float x) const {
        if(!std::isfinite(x)) return x;
        if(x<=-20.f) return 0.f;
        if(x>=20.f) return x;
        const float pos=(x+20.f)*128.f;
        const int index=int(pos);
        // Addition rounding can put the final in-range value at 5120.
        if(index>=5120) return values[5120];
        return values[index]+(pos-index)*(values[index+1]-values[index]);
    }
    void apply(float* data,int count) const {
        int i=0;
#ifdef __ARM_NEON
        for(;i+4<=count;i+=4) {
            const float32x4_t x=vld1q_f32(data+i);
            const uint32x4_t finite=vcleq_f32(vabsq_f32(x),vdupq_n_f32(std::numeric_limits<float>::max()));
            const float32x4_t safe=vbslq_f32(finite,x,vdupq_n_f32(0.f));
            const float32x4_t clipped=vmaxq_f32(vdupq_n_f32(-20.f),vminq_f32(safe,vdupq_n_f32(20.f)));
            const float32x4_t pos=vmulq_n_f32(vaddq_f32(clipped,vdupq_n_f32(20.f)),128.f);
            const int32x4_t ix=vminq_s32(vcvtq_s32_f32(pos),vdupq_n_s32(5119));
            int32_t index[4];vst1q_s32(index,ix);
            const float32x2_t a=vld1_f32(&values[index[0]]),b=vld1_f32(&values[index[1]]);
            const float32x2_t c=vld1_f32(&values[index[2]]),d=vld1_f32(&values[index[3]]);
            const float32x4x2_t pair=vtrnq_f32(vcombine_f32(a,c),vcombine_f32(b,d));
            float32x4_t result=vaddq_f32(pair.val[0],vmulq_f32(vsubq_f32(pos,vcvtq_f32_s32(ix)),vsubq_f32(pair.val[1],pair.val[0])));
            result=vbslq_f32(vcleq_f32(x,vdupq_n_f32(-20.f)),vdupq_n_f32(0.f),result);
            result=vbslq_f32(vcgeq_f32(x,vdupq_n_f32(20.f)),x,result);
            result=vbslq_f32(finite,result,x);
            vst1q_f32(data+i,result);
        }
#endif
        for(;i<count;++i) data[i]=(*this)(data[i]);
    }
};
