#pragma once
#include "stream_packet.h"
#include "stream_output_affine.h"
#include "stream_region_coeff.h"
#include <cmath>
#include <cstring>

// Shared scalar correctness path for a future region dispatcher. Inputs and
// model constants are private/validated buffers. No device or DMA operations.
// HLS synthesis, resource use, timing and board acceptance are still required.
struct RegionByteDestination {
    int8_t* data;
    void set(size_t index,int8_t value) {data[index]=value;}
};
struct RegionWordDestination {
    uint64_t* data;
    void set(size_t index,int8_t value) {
        const unsigned shift=unsigned(index%8)*8;
        const uint64_t previous=index%8?data[index/8]:0;
        data[index/8]=previous|(uint64_t(uint8_t(value))<<shift);
    }
};
template<class Destination> inline int region_prepare_boundary_to(const float* input,size_t elements,
        const float* weight_scale,const float* bias,const float* outgoing,
        size_t channels,uint64_t* packet,size_t packet_bytes,
        Destination quantized,size_t quantized_bytes) {
#ifndef __SYNTHESIS__
    if(!input || !weight_scale || !bias || !outgoing || !packet || !quantized.data)return -1;
#endif
    if(packet_bytes<64 || packet_bytes>STREAM_MAX_PACKET_BYTES || packet_bytes%64)return -1;
    uint32_t header[8];
    for(int i=0;i<4;++i){header[i*2]=uint32_t(packet[i]);header[i*2+1]=uint32_t(packet[i]>>32);}
    if(header[0]!=STREAM_PACKET_MAGIC || header[1]!=STREAM_FP32_PACKET_VERSION ||
       header[2]<1 || header[2]>6 || header[5]!=packet_bytes ||
       header[3]<32+header[2]*48 || header[3]%64 || header[4]%64 ||
       header[4]<header[3] || header[4]>packet_bytes)return -2;
    int fields[12];
    for(int i=0;i<6;++i){fields[i*2]=int32_t(packet[4+i]);fields[i*2+1]=int32_t(packet[4+i]>>32);}
    StreamOp first={fields[0],fields[1],fields[2],fields[3],fields[4],fields[5],fields[6],fields[7],fields[8],fields[9],fields[10],fields[11]};
    if(!stream_op_valid(first) || elements!=size_t(first.ci)*first.h*first.w ||
       header[6]!=elements || quantized_bytes<elements || channels!=size_t(first.co) ||
       uint64_t(header[3])+(uint64_t(first.requant_offset)+first.co)*16>header[4])return -3;
    const size_t plane=size_t(first.h)*first.w;
    float scales[320]={};StreamRequant coefficients[320];StreamOutputAffine affine[320];
    // Validate complete input/constants before modifying either output buffer.
    for(int c=0;c<first.ci;++c) {
        float maximum=0;
        for(size_t p=0;p<plane;++p) {
            const float value=input[size_t(c)*plane+p];
            if(!std::isfinite(value))return -4;
            const float magnitude=std::fabs(value);
            if(magnitude>maximum)maximum=magnitude;
        }
        const int slot=first.kind?c:0;
        if(maximum>scales[slot])scales[slot]=maximum;
    }
    for(int i=0;i<(first.kind?first.ci:1);++i) {
        scales[i]=scales[i]?scales[i]/127.f:1.f;
        if(!std::isfinite(scales[i]) || scales[i]<=0)return -5;
    }
    for(int c=0;c<first.co;++c) {
        const float incoming=scales[first.kind?c:0];
        if(!std::isfinite(weight_scale[c]) || weight_scale[c]<=0 || !std::isfinite(bias[c]))return -6;
        if(header[2]==1) {
            affine[c]={incoming,weight_scale[c],bias[c],0};
            if(!stream_output_affine_valid(affine[c]))return -7;
        } else {
            if(!std::isfinite(outgoing[c]) || outgoing[c]<=0)return -8;
            const double ratio=double(incoming)*double(weight_scale[c])/double(outgoing[c]);
            if(!std::isfinite(ratio) || ratio<0)return -8;
            int shift=region_affine_shift(ratio);
            while(shift>0 && region_affine_scale(ratio,shift)>8388607.0)--shift;
            const double multiplier=region_affine_scale(ratio,shift);
            const double offset=region_affine_scale(double(bias[c])/double(outgoing[c]),shift);
            if(!std::isfinite(multiplier+offset) || multiplier<0 || multiplier>8388607 || std::fabs(offset)>double(int64_t(1)<<60))return -9;
            coefficients[c]={int32_t(region_affine_round(multiplier)),shift,region_affine_round(offset)};
        }
    }
    for(int c=0;c<first.ci;++c)for(size_t p=0;p<plane;++p) {
        float value=input[size_t(c)*plane+p]/scales[first.kind?c:0];
        if(value>127.f)value=127.f;
        if(value<-127.f)value=-127.f;
        // Round an FP32 quotient ties away without an extra rounded +0.5.
        const float magnitude=std::fabs(value);const int integral=int(magnitude);
        const int rounded=integral+(magnitude-float(integral)>=0.5f?1:0);
        quantized.set(size_t(c)*plane+p,int8_t(value<0?-rounded:rounded));
    }
    const size_t destination=header[3]/8+size_t(first.requant_offset)*2;
    for(size_t c=0;c<channels;++c) {
        if(header[2]==1) {
            union FloatBits {float value;uint32_t bits;};
            FloatBits incoming,weight,bias_bits;
            incoming.value=affine[c].incoming;weight.value=affine[c].weight;bias_bits.value=affine[c].bias;
            packet[destination+c*2]=uint64_t(incoming.bits)|(uint64_t(weight.bits)<<32);
            packet[destination+c*2+1]=bias_bits.bits;
        } else {
            packet[destination+c*2]=uint64_t(uint32_t(coefficients[c].multiplier))|(uint64_t(uint32_t(coefficients[c].shift))<<32);
            packet[destination+c*2+1]=uint64_t(coefficients[c].bias);
        }
    }
    return 0;
}
inline int region_prepare_boundary(const float* input,size_t elements,
        const float* weight_scale,const float* bias,const float* outgoing,
        size_t channels,uint64_t* packet,size_t packet_bytes,
        int8_t* quantized,size_t quantized_bytes) {
    return region_prepare_boundary_to(input,elements,weight_scale,bias,outgoing,
        channels,packet,packet_bytes,RegionByteDestination{quantized},quantized_bytes);
}
inline int region_prepare_boundary_packed(const float* input,size_t elements,
        const float* weight_scale,const float* bias,const float* outgoing,
        size_t channels,uint64_t* packet,size_t packet_bytes,
        uint64_t* quantized,size_t quantized_bytes) {
    return region_prepare_boundary_to(input,elements,weight_scale,bias,outgoing,
        channels,packet,packet_bytes,RegionWordDestination{quantized},quantized_bytes);
}
