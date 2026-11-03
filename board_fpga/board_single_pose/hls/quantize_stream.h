#pragma once
#include <stdint.h>

// b[0..63]: 128 little-endian uint32 thresholds. FP32 input starts at
// source_word; INT8 output at output_word, strictly after the input region.
extern "C" int pose_quantize_stream(uint64_t* b,int count,int source_word,int output_word);
