#pragma once
#include "stream_fp32_wire.h"

// Version2 is explicit: never reinterpret a legacy packet as FP32 output.
inline void stream_patch_fp32(std::vector<uint64_t>& packet,
        const std::vector<StreamOp>& ops,const std::vector<StreamRequant>& coeff,
        const std::vector<StreamOutputAffine>& affine) {
    if(packet.size()<8 || packet.size()>STREAM_MAX_PACKET_BYTES/8 || ops.empty())
        throw std::runtime_error("FP32 prepared packet size");
    uint32_t h[8];memcpy(h,packet.data(),sizeof(h));
    const auto& last=ops.back();
    if(h[0]!=STREAM_PACKET_MAGIC || h[1]!=STREAM_FP32_PACKET_VERSION || h[2]!=ops.size() ||
       h[5]!=packet.size()*8 || h[3]%64 || h[4]%64 || h[3]<32+h[2]*sizeof(StreamOp) ||
       h[3]>h[4] || h[4]>h[5] || last.requant_offset<0 ||
       affine.size()!=size_t(last.co) || coeff.size()>size_t(last.requant_offset) ||
       (size_t(last.requant_offset)+affine.size())*16>h[4]-h[3])
        throw std::runtime_error("FP32 prepared coefficient range");
    for(const auto& value:affine)if(!stream_output_affine_valid(value))
        throw std::runtime_error("FP32 affine outside hardware contract");
    char* data=reinterpret_cast<char*>(packet.data());
    if(!coeff.empty())memcpy(data+h[3],coeff.data(),coeff.size()*16);
    memcpy(data+h[3]+size_t(last.requant_offset)*16,affine.data(),affine.size()*16);
}
