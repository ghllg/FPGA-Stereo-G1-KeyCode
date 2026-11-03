#pragma once
#include <cmath>
#include <cstring>
#include <stdint.h>
#include <stdexcept>

// For normal bounded scales, turn IEEE float division + half-away rounding
// into 127 exact integer comparisons. Positive IEEE bit patterns are ordered.
// Each half-integer t has an even significand. Round-to-nearest division first
// reaches t at the midpoint of t and its predecessor (tie rounds up to t).
// The scale (24 bits) times that midpoint (<=25 bits) is exact in binary64.
inline void pose_make_thresholds(float scale,uint32_t table[128]) {
    if (!(scale>=1e-30f && scale<=1e30f))
        throw std::runtime_error("Threshold quantization requires scale in [1e-30,1e30]");
    for (int n=1;n<=127;++n) {
        const float half=float(n)-0.5f;
        uint32_t half_bits;std::memcpy(&half_bits,&half,4);--half_bits;
        float previous;std::memcpy(&previous,&half_bits,4);
        const double midpoint=(double(half)+double(previous))*0.5;
        const double exact=double(scale)*midpoint;
        // Old Windows HLS GCC may retain x87 excess precision after a float
        // cast. Materialize binary32 before deciding whether ceil needs 1 ULP.
        volatile float rounded=float(exact);
        float boundary=rounded;
        uint32_t bits;std::memcpy(&bits,&boundary,4);
        if (double(boundary)<exact) ++bits;
        table[n-1]=bits;
    }
    table[127]=0x7f800000u;
}
inline int8_t pose_threshold_quantize_bits(uint32_t value,const uint32_t table[128]) {
    const uint32_t magnitude=value&0x7fffffffu;
    unsigned index=0;
    for (int bit=6;bit>=0;--bit) {
        const unsigned candidate=index|(1u<<bit);
        if (magnitude>=table[candidate-1]) index=candidate;
    }
    return int8_t(value>>31 ? -int(index) : int(index));
}
