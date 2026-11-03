#pragma once
#include "stream_wire.h"
#include "../hls/stream_output_affine.h"

// Build a distinct version2 packet; legacy version1 remains unchanged.
inline std::vector<uint64_t> stream_pack_fp32_output(const std::vector<StreamOp>& ops,
        const std::vector<int8_t>& weights,const std::vector<StreamRequant>& coeff,
        const std::vector<StreamOutputAffine>& output) {
    if(ops.empty() || output.size()!=size_t(ops.back().co))
        throw std::runtime_error("FP32 output coefficient count");
    const auto& last=ops.back();
    if(last.requant_offset<0 || size_t(last.requant_offset)>6*320)
        throw std::runtime_error("FP32 output coefficient offset");
    for(const auto& q:output)if(!stream_output_affine_valid(q))
        throw std::runtime_error("FP32 output coefficient outside the bounded hardware contract");
    std::vector<StreamRequant> slots=coeff;
    if(slots.size()>size_t(last.requant_offset))
        throw std::runtime_error("FP32 output aliases internal requant coefficients");
    slots.resize(size_t(last.requant_offset)+output.size());
    auto packet=stream_pack(ops,weights,slots);
    uint32_t header[8];memcpy(header,packet.data(),sizeof(header));
    header[1]=STREAM_FP32_PACKET_VERSION;
    memcpy(packet.data(),header,sizeof(header));
    memcpy(reinterpret_cast<char*>(packet.data())+header[3]+size_t(last.requant_offset)*16,
           output.data(),output.size()*16);
    return packet;
}
