#pragma once
#ifdef __SYNTHESIS__
#include <ap_shift_reg.h>
template<class T,unsigned Depth> using PoseFeedbackShift=ap_shift_reg<T,Depth>;
#else
// Native model only. Hardware uses the vendor's SRL inference primitive;
// its implementation is supplied by the installed HLS tool, not this repo.
template<class T,unsigned Depth> class PoseFeedbackShift {
    T values[Depth]={};
    unsigned oldest=0;
public:
    T read() const {return values[oldest];}
    T shift(T value) {
        const T result=values[oldest];values[oldest]=value;
        oldest=(oldest+1)%Depth;return result;
    }
};
#endif
