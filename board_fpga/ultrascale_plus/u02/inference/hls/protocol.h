#pragma once
#include "conv_contract.h"
enum { LIGHT_DW_FLAG=0x20000000, LIGHT_DW_MAX_PIXELS=3072 };
inline bool light_dw_valid(const PoseConv& p,int channels,int count,int kernel) {
    return channels==p.channels && channels>0 && channels<=512 &&
        (p.kw==1 || p.kw==3) && (p.kh==1 || p.kh==3) && kernel==p.kw*p.kh &&
        pose_conv_valid(p,count,channels*kernel) && p.w*p.h<=LIGHT_DW_MAX_PIXELS &&
        count<=LIGHT_DW_MAX_PIXELS;
}
