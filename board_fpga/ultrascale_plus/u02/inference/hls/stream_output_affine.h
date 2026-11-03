#pragma once
#include <stdint.h>

// Separate wire version: the final requant-offset slots contain these FP32
// coefficients. All earlier slots retain the original integer requant ABI.
static const uint32_t STREAM_FP32_PACKET_VERSION=2;
struct StreamOutputAffine {float incoming,weight,bias;uint32_t reserved;};
static_assert(sizeof(StreamOutputAffine)==16,"Output affine wire ABI");

// Conservative model contract for the proposed hardware FP32 path. A final
// int8 convolution sum has magnitude <=320*128*128 <2^23. The two separately
// rounded multiplies stay between 2^-80 and 2^103 for nonzero values. A finite
// bias in this range cannot cancel to a subnormal FP32 value (spacing>=2^-103).
// Actual HLS/RTL still must prove the operation order and bit-exact rounding.
inline bool stream_output_affine_valid(const StreamOutputAffine& q) {
    const float max_scale=1099511627776.f,min_scale=1.f/max_scale;
    const float max_bias=1208925819614629174706176.f,min_bias=1.f/max_bias;
    const float magnitude=q.bias<0.f?-q.bias:q.bias;
    return q.reserved==0 && q.incoming>=min_scale && q.incoming<=max_scale &&
           q.weight>=min_scale && q.weight<=max_scale &&
           (q.bias==0.f || (magnitude>=min_bias && magnitude<=max_bias));
}
