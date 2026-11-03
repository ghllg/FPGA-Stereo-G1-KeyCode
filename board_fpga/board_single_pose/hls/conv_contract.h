#pragma once
#include "pose_gemm.h"

// Same AXI-Lite ABI as GEMM. A high bit in K requests a 64-byte input
// descriptor followed by planar INT8 CHW pixels instead of im2col B[K,N].
// Older GEMM firmware rejects this K as out of range, never silently runs it.
enum { POSE_CONV_FLAG=0x40000000, POSE_CONV_MAGIC=0x43564e31, POSE_CONV_WORDS=16 };
struct PoseConv {
    int w,h,channels,kw,kh,sw,sh,left,top,ow,oh,start;
};
// A pointwise stride-1 convolution only depends on the flattened spatial
// index. Fold wide decoder tensors (e.g. 2100x4) into an equivalent <=320
// rectangle without copying data or changing channel/weight ordering.
inline bool pose_conv_fold_pointwise(PoseConv& p) {
    if (p.kw!=1 || p.kh!=1 || p.sw!=1 || p.sh!=1 || p.left || p.top ||
        p.w<1 || p.h<1 || p.ow!=p.w || p.oh!=p.h || int64_t(p.w)*p.h>320*320)
        return false;
    const int pixels=p.w*p.h;
    for (int w=pixels<320?pixels:320;w>0;--w) {
        if (pixels%w==0 && pixels/w<=320) {
            p.w=p.ow=w;p.h=p.oh=pixels/w;return true;
        }
    }
    return false;
}
inline bool pose_conv_valid(const PoseConv& p,int n,int k) {
    return p.w>0 && p.w<=320 && p.h>0 && p.h<=320 &&
        p.channels>0 && p.channels<=512 && (p.kw==1 || p.kw==3) &&
        (p.kh==1 || p.kh==3) && (p.sw==1 || p.sw==2) &&
        (p.sh==1 || p.sh==2) && p.left>=0 && p.left<=1 &&
        p.top>=0 && p.top<=1 && p.ow>0 && p.ow<=320 &&
        p.oh>0 && p.oh<=320 && p.start>=0 && n>0 && n<=POSE_MAX_N &&
        p.start<=p.ow*p.oh && n<=p.ow*p.oh-p.start &&
        k==p.channels*p.kw*p.kh && k<=POSE_MAX_K;
}
inline void pose_conv_header(uint32_t* d,const PoseConv& p) {
    d[0]=POSE_CONV_MAGIC;d[1]=p.w;d[2]=p.h;d[3]=p.channels;
    d[4]=p.kw;d[5]=p.kh;d[6]=p.sw;d[7]=p.sh;d[8]=p.left;
    d[9]=p.top;d[10]=p.ow;d[11]=p.oh;d[12]=p.start;
    d[13]=d[14]=d[15]=0;
}
extern "C" int pose_conv_reuse(const uint64_t* a,const uint64_t* b,int32_t* c,int m,int n,int k);
