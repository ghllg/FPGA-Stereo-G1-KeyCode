#pragma once
#include "stream_region_execute.h"
#include "stream_region_model_packet.h"
#include "stream_region_boundary.h"
#include "region_shared_compute.h"

// Stable, driver-owned model memory is required throughout the command.
// One shared physical compute function, one packet and one quantized staging
// tensor. Scratch remains external DDR; no nine-engine replication.
class RegionHlsContext {
    unsigned groups;
    bool lookup(const unsigned char* model,unsigned id,RegionModelGroup& found) {
        for(unsigned i=0;i<groups;++i) {
            RegionModelGroup g=region_model_group(model,64+i*64);
            if(g.id==id){found=g;return true;}
        }
        return false;
    }
public:
    uint64_t packet[REGION_MODEL_MAX_PACKET/8];
    uint64_t quantized[12288];
    float weight[320],bias[320],outgoing[320];
    unsigned group_executions;
    RegionHlsContext(unsigned count):groups(count),group_executions(0){}
    bool matches(const unsigned char* model,unsigned id,const RegionTensor& input,const RegionTensor& output,unsigned scope) {
        RegionModelGroup g;if(!lookup(model,id,g))return false;
        const uint32_t count=region_model_word(model,g.packet_offset+8);
        StreamOp first=region_model_op(model,g.packet_offset+32);
        StreamOp last=region_model_op(model,g.packet_offset+32+(count-1)*48);
        return scope==unsigned(first.kind) && input.c==unsigned(first.ci) && input.h==unsigned(first.h) &&
            input.w==unsigned(first.w) && output.c==unsigned(last.co) && output.h==unsigned(last.oh) && output.w==unsigned(last.ow);
    }
    int run(const unsigned char* model,unsigned id,const float* input,float* output,unsigned scope) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION instances=region_shared_compute limit=1 function
#pragma HLS RESOURCE variable=packet core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=quantized core=RAM_2P_URAM
        RegionModelGroup g;if(!lookup(model,id,g))return -30;
        for(unsigned i=0;i<g.packet_bytes/8;++i) {
            packet[i]=uint64_t(region_model_word(model,g.packet_offset+i*8))|
                (uint64_t(region_model_word(model,g.packet_offset+i*8+4))<<32);
        }
        uint32_t h[8];
        for(int i=0;i<4;++i){h[i*2]=uint32_t(packet[i]);h[i*2+1]=uint32_t(packet[i]>>32);}
        if(scope!=uint32_t(packet[4]))return -31;
        for(unsigned i=0;i<g.channels;++i) {
            weight[i]=region_model_float(model,g.weight_scale_offset+i*4);
            bias[i]=region_model_float(model,g.bias_offset+i*4);
            outgoing[i]=region_model_float(model,g.outgoing_offset+i*4);
        }
        int status=region_prepare_boundary_packed(input,h[6],weight,bias,outgoing,g.channels,
            packet,g.packet_bytes,quantized,sizeof(quantized));
        if(status)return status;
        status=region_shared_compute(packet,quantized,output,h[2],g.packet_bytes,STREAM_PACKET_MAGIC);
        if(!status)++group_executions;
        return status;
    }
};
