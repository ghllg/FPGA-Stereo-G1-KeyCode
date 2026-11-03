#pragma once
#include "stream_contract.h"
static const uint32_t STREAM_PACKET_MAGIC=0x55303253u;
static const uint32_t STREAM_PACKET_VERSION=1;
static const int STREAM_MAX_OPS=6;
static const int STREAM_MAX_PACKET_BYTES=1048576;
static const int STREAM_MAX_WEIGHT_BYTES=102400;
// Four 64-bit header words, then n * 48-byte StreamOp records.
// Header u32 fields: magic, version, n, coefficients byte offset,
// weights byte offset, total byte size, input byte size, output INT32 count.
// Coefficients and each operator's weights are aligned to 64 bytes.
static_assert(sizeof(StreamOp)==48,"Stream descriptor ABI");
