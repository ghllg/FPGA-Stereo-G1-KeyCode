#pragma once
#include <stdint.h>
#include <memory>
#include <string>
#include <stdexcept>
#include "../hls/conv_contract.h"

struct Backend {
    uint64_t calls=0, macs=0;
    double prepare_ms=0, wait_ms=0, readback_ms=0;
    uint64_t input_bytes=0, output_bytes=0;
    bool cached_dma=false;
    bool directional_dma=false;
    double custom_conv_ms=0, quantize_ms=0, im2col_ms=0, dequantize_ms=0;
    uint64_t conv_layers=0;
    uint64_t cpu_selected_layers=0;
    double cpu_selected_ms=0;
    uint64_t direct_calls=0;
    uint64_t convolution_calls=0,resident_reuses=0;
    uint64_t pl_quantize_calls=0;
    virtual ~Backend() {}
    virtual void run(const int8_t* a, const int8_t* b, int32_t* c, int m, int n, int k)=0;
    // Optional borrowed DMA storage. Input is writable until run_prepared;
    // returned results remain valid only until the next backend operation.
    // nullptr means the caller must use run with its own buffers.
    virtual int8_t* prepare_input(int m,int n,int k) {return nullptr;}
    virtual const int32_t* run_prepared(const int8_t* a) {
        throw std::runtime_error("Prepared input unsupported by this backend");
    }
    // first=true starts a new immutable weight/image pair. Later chunks may
    // change only start and N; all other fields and input storage stay valid.
    virtual const int32_t* run_convolution(const int8_t* a,const int8_t* image,
        int32_t* scratch,int m,int n,int k,const PoseConv& shape,bool first) {
        throw std::runtime_error("Direct convolution unsupported by this backend");
    }
    virtual bool hardware() const=0;
    virtual const int32_t* run_quantized_convolution(const int8_t* a,const float* image,
        size_t channel_stride,const uint32_t* thresholds,int32_t* scratch,
        int m,int n,int k,const PoseConv& shape,bool first) {
        throw std::runtime_error("PL quantization unsupported by this backend");
    }
};
std::unique_ptr<Backend> make_backend(const std::string& mode,
    const std::string& uio="/dev/uio0", const std::string& dma="udmabuf0");
