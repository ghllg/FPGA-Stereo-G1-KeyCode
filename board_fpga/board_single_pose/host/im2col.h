#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>

// One channel/kernel offset, restricted to an output interval. Copy interior
// row spans without repeating boundary checks for every output pixel.
inline void pose_column(const int8_t* src, int8_t* dst, int iw, int ih,
                        int ow, int start, int n, int sw, int sh,
                        int pad_left, int pad_top, int dx, int dy) {
    int done=0;
    while(done<n) {
        const int at=start+done, ox=at%ow, oy=at/ow;
        const int span=std::min(n-done,ow-ox);
        const int y=oy*sh-pad_top+dy, x=ox*sw-pad_left+dx;
        if(y<0 || y>=ih) std::memset(dst+done,0,span);
        else {
            const int lo=std::min(span,std::max(0,(-x+sw-1)/sw));
            const int hi=std::max(lo,std::min(span,x>=iw?0:(iw-1-x)/sw+1));
            std::memset(dst+done,0,lo);
            if(hi>lo) {
                const int8_t* row=src+y*iw+x+lo*sw;
                if(sw==1) std::memcpy(dst+done+lo,row,hi-lo);
                else for(int j=lo;j<hi;++j) dst[done+j]=row[(j-lo)*sw];
            }
            std::memset(dst+done+hi,0,span-hi);
        }
        done+=span;
    }
}
