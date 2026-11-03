#pragma once
#include <stdint.h>

// Portable command/math contract shared by the Host reference and future HLS.
// INT32 is intentional: the full model exceeds the V13A subgraph's 20-bit sum.
struct StreamOp {
    int kind, ci, co, h, w, oh, ow, stride, pad, relu;
    int weight_offset, requant_offset;
};
struct StreamRequant { int32_t multiplier, shift; int64_t bias; };
static_assert(sizeof(StreamRequant)==16,"Stream requant ABI");
inline int8_t stream_requant(int32_t sum,const StreamRequant& q,bool relu) {
    const int64_t value=int64_t(sum)*q.multiplier+q.bias;
    const uint64_t magnitude=value<0?uint64_t(-value):uint64_t(value);
    const uint64_t rounded=(magnitude+(q.shift?(uint64_t(1)<<(q.shift-1)):0))>>q.shift;
    const int64_t signed_value=value<0?-int64_t(rounded):int64_t(rounded);
    return int8_t(signed_value>127?127:(signed_value<(relu?0:-127)?(relu?0:-127):signed_value));
}
inline bool stream_op_valid(const StreamOp& op) {
    return (op.kind==0 || op.kind==1) && op.ci>0 && op.ci<=320 && op.co>0 && op.co<=320 &&
        op.h>0 && op.h<=64 && op.w>0 && op.w<=48 && op.oh>0 && op.ow>0 &&
        (op.stride==1 || op.stride==2) && op.pad>=0 && op.pad<=1 &&
        (op.relu==0 || op.relu==1) && op.weight_offset>=0 && op.requant_offset>=0 &&
        (op.kind==0?(op.stride==1 && op.pad==0):(op.ci==op.co)) &&
        op.oh==(op.h+2*op.pad-(op.kind?3:1))/op.stride+1 &&
        op.ow==(op.w+2*op.pad-(op.kind?3:1))/op.stride+1 &&
        op.ci*op.h*op.w<=98304 && op.co*op.oh*op.ow<=98304;
}
