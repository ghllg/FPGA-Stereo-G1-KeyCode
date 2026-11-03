#ifndef U02_STREAM_API_H
#define U02_STREAM_API_H
#include "u02_pose_api.h"
#define U02_STREAM_ABI 2u
#define U02_STREAM_MAGIC 0x55303253u
#define U02_STREAM_PACKET_BYTES 1048576u
#define U02_STREAM_INPUT_BYTES 98304u
#define U02_STREAM_OUTPUT_BYTES 393216u
#define U02_STREAM_B_OFFSET U02_STREAM_PACKET_BYTES
#define U02_STREAM_C_OFFSET (U02_STREAM_PACKET_BYTES + U02_STREAM_INPUT_BYTES)
#define U02_STREAM_REG_RETURN 0x10
#define U02_STREAM_REG_A 0x18
#define U02_STREAM_REG_B 0x24
#define U02_STREAM_REG_C 0x30
#define U02_STREAM_REG_M 0x3c
#define U02_STREAM_REG_N 0x44
#define U02_STREAM_REG_K 0x4c
struct u02_stream_op {
    __s32 kind, ci, co, h, w, oh, ow, stride, pad, relu;
    __s32 weight_offset, requant_offset;
};
struct u02_stream_requant {__s32 multiplier, shift; __s64 bias;};
#endif
