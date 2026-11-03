#pragma once
#include "../hls/stream_packet.h"
#include <cstring>
#include <vector>
#include <stdexcept>

inline std::vector<uint64_t> stream_pack(const std::vector<StreamOp>& source,
                                        const std::vector<int8_t>& weights,
                                        const std::vector<StreamRequant>& coeff) {
    if(source.empty() || source.size()>STREAM_MAX_OPS) throw std::runtime_error("Stream operator count");
    std::vector<StreamOp> ops=source;size_t packed_weight_bytes=0;
    for(auto& op:ops) {
        const size_t count=size_t(op.co)*(op.kind?9:op.ci);
        if(!stream_op_valid(op) || size_t(op.weight_offset)+count>weights.size()) throw std::runtime_error("Stream weight range");
        const size_t offset=(packed_weight_bytes+63)&~size_t(63);
        packed_weight_bytes=offset+count;op.weight_offset=int(offset);
    }
    const size_t qbase=(32+ops.size()*sizeof(StreamOp)+63)&~size_t(63);
    const size_t wbase=(qbase+coeff.size()*sizeof(StreamRequant)+63)&~size_t(63);
    const size_t bytes=(wbase+packed_weight_bytes+63)&~size_t(63);
    if(bytes>STREAM_MAX_PACKET_BYTES) throw std::runtime_error("Stream packet too large");
    std::vector<uint64_t> packet(bytes/8,0);
    const auto& first=ops.front();const auto& last=ops.back();
    uint32_t header[8]={STREAM_PACKET_MAGIC,STREAM_PACKET_VERSION,uint32_t(ops.size()),uint32_t(qbase),
                       uint32_t(wbase),uint32_t(bytes),uint32_t(first.ci*first.h*first.w),uint32_t(last.co*last.oh*last.ow)};
    char* dst=reinterpret_cast<char*>(packet.data());
    memcpy(dst,header,sizeof(header));memcpy(dst+32,ops.data(),ops.size()*sizeof(StreamOp));
    if(!coeff.empty())memcpy(dst+qbase,coeff.data(),coeff.size()*sizeof(StreamRequant));
    // Fill the final aligned packet directly: no growing temporary weight
    // vector or second whole-weight copy on every network submission.
    for(size_t i=0;i<ops.size();++i) {
        const size_t count=size_t(ops[i].co)*(ops[i].kind?9:ops[i].ci);
        memcpy(dst+wbase+ops[i].weight_offset,weights.data()+source[i].weight_offset,count);
    }
    return packet;
}

// Update only runtime affine coefficients in a packet whose operators and
// weights were packed once at model load. Padding and weights stay untouched.
inline void stream_patch_requant(std::vector<uint64_t>& packet,
                                 const std::vector<StreamRequant>& coeff) {
    if(packet.size()<8 || packet.size()>STREAM_MAX_PACKET_BYTES/8)
        throw std::runtime_error("Prepared stream packet size");
    uint32_t header[8];memcpy(header,packet.data(),sizeof(header));
    if(header[0]!=STREAM_PACKET_MAGIC || header[1]!=STREAM_PACKET_VERSION ||
       !header[2] || header[2]>STREAM_MAX_OPS || header[5]!=packet.size()*8 ||
       header[3]%64 || header[4]%64 || header[3]<32+header[2]*sizeof(StreamOp) ||
       header[3]>header[4] || header[4]>header[5] ||
       coeff.size()>(header[4]-header[3])/sizeof(StreamRequant))
        throw std::runtime_error("Prepared stream coefficient range");
    if(!coeff.empty())memcpy(reinterpret_cast<char*>(packet.data())+header[3],
                            coeff.data(),coeff.size()*sizeof(StreamRequant));
}
