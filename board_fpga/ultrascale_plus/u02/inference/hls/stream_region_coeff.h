#pragma once
#include <stdint.h>

// IEEE binary64 arithmetic for the validated region coefficient domain.
// Avoid host STL/libm dependencies in the 2018.3 synthesis frontend.
union RegionDoubleBits {double value;uint64_t bits;};
inline int region_affine_shift(double ratio) {
    RegionDoubleBits encoded;encoded.value=ratio;
    const int exponent=int((encoded.bits>>52)&2047);
    // Zero/subnormal positive ratios already saturate to the maximum shift.
    const int shift=exponent?23-(exponent-1022):40;
    return shift<0?0:(shift>40?40:shift);
}
inline double region_affine_scale(double value,int shift) {
    RegionDoubleBits factor;factor.bits=uint64_t(1023+shift)<<52;
    return value*factor.value;
}
inline int64_t region_affine_round(double value) {
    const double magnitude=value<0?-value:value;
    const int64_t integral=int64_t(magnitude);
    const int64_t rounded=integral+(magnitude-double(integral)>=0.5?1:0);
    return value<0?-rounded:rounded;
}
