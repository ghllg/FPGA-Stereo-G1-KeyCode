#pragma once
#include "protocol.h"

enum { LIGHT_PAIR_FLAG=0x10000000, LIGHT_PAIR_MAX_VALUES=98304 };
struct LightAffine { int32_t multiplier,shift;int64_t bias; };
typedef char LightAffineMustBe16Bytes[(sizeof(LightAffine)==16)?1:-1];
inline bool light_affine_valid(const LightAffine& a) {
    return a.multiplier>=0 && a.shift>=1 && a.shift<=30 &&
        a.bias>=-(int64_t(1)<<60) && a.bias<=(int64_t(1)<<60);
}
// ReLU followed by positive INT8 quantization. All values are bounded so
// multiply + bias + rounding cannot overflow signed 64-bit arithmetic.
inline int8_t light_affine_relu(int32_t sum,const LightAffine& a) {
    const int64_t product=int64_t(sum)*a.multiplier;
#pragma HLS RESOURCE variable=product core=MulnS latency=3
    const int64_t v=product+a.bias;
    if(v<=0)return 0;
    const int64_t q=(v+(int64_t(1)<<(a.shift-1)))>>a.shift;
    return int8_t(q>127?127:q);
}
inline bool light_pair_valid(const PoseConv& dw,int input_channels,int output_pixels) {
    // Bound first, then use narrow products. Full 32-bit descriptor products
    // otherwise spend many DSPs on validation rather than CNN arithmetic.
    if(dw.w<1 || dw.w>320 || dw.h<1 || dw.h>320 || dw.channels<1 || dw.channels>512 ||
       input_channels<1 || input_channels>512 || dw.ow<1 || dw.ow>320 || dw.oh<1 || dw.oh>320 ||
       (dw.kw!=1 && dw.kw!=3) || (dw.kh!=1 && dw.kh!=3) ||
       (dw.sw!=1 && dw.sw!=2) || (dw.sh!=1 && dw.sh!=2) ||
       dw.left<0 || dw.left>1 || dw.top<0 || dw.top>1 || dw.start!=0)return false;
    const uint32_t pixels=uint16_t(dw.w)*uint16_t(dw.h),out=uint16_t(dw.ow)*uint16_t(dw.oh);
    if(pixels>LIGHT_DW_MAX_PIXELS || out>LIGHT_DW_MAX_PIXELS || output_pixels!=int(out))return false;
    const int nw=dw.w+2*dw.left-dw.kw,nh=dw.h+2*dw.top-dw.kh;
    if(nw<0 || nh<0 || dw.ow!=(dw.sw==2?nw/2:nw)+1 || dw.oh!=(dw.sh==2?nh/2:nh)+1)return false;
    return uint16_t(pixels)*uint16_t(input_channels)<=LIGHT_PAIR_MAX_VALUES &&
           uint16_t(pixels)*uint16_t(dw.channels)<=LIGHT_PAIR_MAX_VALUES;
}
inline int light_align64(int n) {return (n+63)&~63;}
// Offsets in bytes, relative to the three AXI master base pointers.
struct LightPairLayout {
    int dw_weights,pw_features,dw_features,mid_image,feature_bytes,scratch,result_bytes;
    LightPairLayout():dw_weights(0),pw_features(0),dw_features(0),mid_image(0),feature_bytes(0),scratch(0),result_bytes(0) {}
    LightPairLayout(int input_channels,const PoseConv& dw) {
        const uint16_t pixels=uint16_t(dw.w*dw.h),channels=uint16_t(dw.channels);
        const uint32_t weight_values=channels*uint16_t(input_channels);
        const uint32_t input_values=uint16_t(input_channels)*pixels;
        const uint32_t mid_values=channels*pixels;
        const uint32_t output_values=channels*uint16_t(dw.ow*dw.oh);
#pragma HLS RESOURCE variable=weight_values core=MulnS latency=2
#pragma HLS RESOURCE variable=input_values core=MulnS latency=2
#pragma HLS RESOURCE variable=mid_values core=MulnS latency=2
#pragma HLS RESOURCE variable=output_values core=MulnS latency=2
        dw_weights=light_align64(weight_values);
        pw_features=64+light_align64(channels*16);
        dw_features=pw_features+64+light_align64(input_values);
        mid_image=dw_features+64;
        feature_bytes=mid_image+light_align64(mid_values);
        scratch=light_align64(output_values*4);
        result_bytes=scratch+light_align64(mid_values*4);
    }
};
