#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

// Optional exact coefficient preparation for the complete-network Host.
// Fall back to libm for subnormal, nonfinite or out-of-range exponents.
inline int stream_affine_shift(double ratio) {
    int exponent=0;
#ifdef U02_STREAM_FAST_AFFINE
    static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,"IEEE binary64 required");
    uint64_t bits;std::memcpy(&bits,&ratio,sizeof(bits));
    const int encoded=int((bits>>52)&2047);
    if(encoded>0 && encoded<2047) exponent=encoded-1022;
    else std::frexp(ratio,&exponent);
#else
    std::frexp(ratio,&exponent);
#endif
    return ratio?std::max(0,std::min(40,23-exponent)):40;
}

inline double stream_affine_scale(double value,int shift) {
#ifdef U02_STREAM_FAST_AFFINE
    uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));
    const int exponent=int((bits>>52)&2047);
    if(shift>=0 && shift<=40 && exponent>0 && exponent+shift<2047) {
        bits=(bits&~(uint64_t(2047)<<52))|(uint64_t(exponent+shift)<<52);
        double result;std::memcpy(&result,&bits,sizeof(result));return result;
    }
#endif
    return std::ldexp(value,shift);
}

// Caller validates finite values and |value| <= 2^60 before conversion.
// FCVTAS has the same nearest/ties-away rule as llround, independent of FPCR.
inline int64_t stream_affine_round(double value) {
#if defined(U02_STREAM_FAST_AFFINE) && defined(__aarch64__)
    return vcvtad_s64_f64(value);
#else
    return std::llround(value);
#endif
}
