#pragma once
#include "../hls/stream_region_execute.h"
#include "../hls/stream_region_model_packet.h"
#include "../hls/stream_region_boundary.h"
#include "stream_reference.h"
#include "stream_dequantize.h"
#include <vector>
#include <stdexcept>

// CPU correctness executor only. A future HLS Context replaces run() with a
// single shared physical kernel. This class never accesses a board device.
class RegionReferenceContext {
    struct Group {unsigned id;std::vector<uint64_t> packet;std::vector<float> weight,bias,outgoing;};
    std::vector<Group> groups;
    Group& lookup(unsigned id) {
        for(auto& g:groups)if(g.id==id)return g;
        throw std::runtime_error("Missing region model group");
    }
public:
    std::vector<int8_t> last_quantized;
    std::vector<uint64_t> last_packet;
    unsigned group_executions=0;
    RegionReferenceContext(const void* raw,size_t bytes) {
        if(region_model_validate(raw,bytes))throw std::runtime_error("Invalid region model bundle");
        const unsigned char* data=static_cast<const unsigned char*>(raw);
        RegionModelHeader h;memcpy(&h,data,64);
        for(unsigned i=0;i<h.groups;++i) {
            RegionModelGroup s;memcpy(&s,data+64+i*64,64);Group g;g.id=s.id;
            g.packet.resize(s.packet_bytes/8);memcpy(g.packet.data(),data+s.packet_offset,s.packet_bytes);
            g.weight.resize(s.channels);g.bias.resize(s.channels);g.outgoing.resize(s.channels);
            memcpy(g.weight.data(),data+s.weight_scale_offset,s.channels*4);
            memcpy(g.bias.data(),data+s.bias_offset,s.channels*4);memcpy(g.outgoing.data(),data+s.outgoing_offset,s.channels*4);
            groups.push_back(g);
        }
    }
    bool matches(unsigned id,const RegionTensor& input,const RegionTensor& output,unsigned scope) {
        Group& g=lookup(id);uint32_t h[8];memcpy(h,g.packet.data(),32);
        StreamOp first,last;const char* raw=reinterpret_cast<const char*>(g.packet.data());
        memcpy(&first,raw+32,48);memcpy(&last,raw+32+(h[2]-1)*48,48);
        return scope==unsigned(first.kind) && input.c==unsigned(first.ci) && input.h==unsigned(first.h) &&
            input.w==unsigned(first.w) && output.c==unsigned(last.co) && output.h==unsigned(last.oh) && output.w==unsigned(last.ow);
    }
    static void compute(const std::vector<uint64_t>& packet,const int8_t* input,float* output) {
        uint32_t h[8];memcpy(h,packet.data(),32);const char* raw=reinterpret_cast<const char*>(packet.data());
        std::vector<StreamOp> ops(h[2]);memcpy(ops.data(),raw+32,ops.size()*48);
        const StreamOp& last=ops.back();
        std::vector<StreamRequant> coeff(last.requant_offset);
        if(!coeff.empty())memcpy(coeff.data(),raw+h[3],coeff.size()*16);
        std::vector<int8_t> weights(packet.size()*8-h[4]);memcpy(weights.data(),raw+h[4],weights.size());
        std::vector<int32_t> sums(h[7]);stream_reference(ops,weights,coeff,input,sums.data());
        for(int c=0;c<last.co;++c) {
            StreamOutputAffine affine;memcpy(&affine,raw+h[3]+(last.requant_offset+c)*16,16);
            if(!stream_output_affine_valid(affine))throw std::runtime_error("Invalid reference affine");
            const int plane=last.oh*last.ow;
            stream_dequantize_array(sums.data()+c*plane,output+c*plane,plane,affine.incoming,affine.weight,affine.bias,last.relu!=0);
        }
    }
    int run(unsigned id,const float* input,float* output,unsigned scope) {
        Group& g=lookup(id);last_packet=g.packet;uint32_t h[8];memcpy(h,last_packet.data(),32);
        StreamOp first;memcpy(&first,reinterpret_cast<char*>(last_packet.data())+32,48);
        if(scope!=unsigned(first.kind))return -30;
        last_quantized.resize(h[6]);
        const int status=region_prepare_boundary(input,h[6],g.weight.data(),g.bias.data(),g.outgoing.data(),g.weight.size(),
            last_packet.data(),last_packet.size()*8,last_quantized.data(),last_quantized.size());
        if(status)return status;
        compute(last_packet,last_quantized.data(),output);++group_executions;
        return 0;
    }
};
