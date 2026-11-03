#pragma once
#include "stream_packet.h"
#include "stream_output_affine.h"
#include <cstring>

static const uint32_t REGION_MODEL_MAGIC=0x4d323055u;
static const unsigned REGION_MODEL_MAX_GROUPS=16,REGION_MODEL_MAX_PACKET=16384;
struct RegionModelHeader {uint32_t magic,version,bytes,groups,table_offset,reserved[11];};
struct RegionModelGroup {uint32_t id,packet_offset,packet_bytes,weight_scale_offset,bias_offset,outgoing_offset,channels,reserved[9];};
static_assert(sizeof(RegionModelHeader)==64,"Region model header ABI");
static_assert(sizeof(RegionModelGroup)==64,"Region model group ABI");
inline uint32_t region_model_word(const unsigned char* data,size_t at) {
    uint32_t result=0;
    for(unsigned i=0;i<4;++i)result|=uint32_t(data[at+i])<<(i*8);
    return result;
}
inline float region_model_float(const unsigned char* data,size_t at) {
    union Bits {float value;uint32_t bits;};Bits result;
    result.bits=region_model_word(data,at);return result.value;
}
inline RegionModelGroup region_model_group(const unsigned char* data,size_t at) {
    RegionModelGroup g;
    g.id=region_model_word(data,at);g.packet_offset=region_model_word(data,at+4);
    g.packet_bytes=region_model_word(data,at+8);g.weight_scale_offset=region_model_word(data,at+12);
    g.bias_offset=region_model_word(data,at+16);g.outgoing_offset=region_model_word(data,at+20);
    g.channels=region_model_word(data,at+24);
    for(unsigned i=0;i<9;++i)g.reserved[i]=region_model_word(data,at+28+i*4);
    return g;
}
inline StreamOp region_model_op(const unsigned char* data,size_t at) {
    StreamOp op={int32_t(region_model_word(data,at)),int32_t(region_model_word(data,at+4)),
        int32_t(region_model_word(data,at+8)),int32_t(region_model_word(data,at+12)),
        int32_t(region_model_word(data,at+16)),int32_t(region_model_word(data,at+20)),
        int32_t(region_model_word(data,at+24)),int32_t(region_model_word(data,at+28)),
        int32_t(region_model_word(data,at+32)),int32_t(region_model_word(data,at+36)),
        int32_t(region_model_word(data,at+40)),int32_t(region_model_word(data,at+44))};
    return op;
}
inline int region_model_validate(const void* raw,size_t bytes) {
#ifndef __SYNTHESIS__
    if(!raw)return -1;
#endif
    if(bytes<128 || bytes>1024*1024 || bytes%64)return -1;
    const unsigned char* data=static_cast<const unsigned char*>(raw);
    RegionModelHeader h;
    h.magic=region_model_word(data,0);h.version=region_model_word(data,4);
    h.bytes=region_model_word(data,8);h.groups=region_model_word(data,12);h.table_offset=region_model_word(data,16);
    for(unsigned i=0;i<11;++i)h.reserved[i]=region_model_word(data,20+i*4);
    if(h.magic!=REGION_MODEL_MAGIC || h.version!=1 || h.bytes!=bytes || !h.groups ||
       h.groups>REGION_MODEL_MAX_GROUPS || h.table_offset!=64 || 64+uint64_t(h.groups)*64>bytes)return -2;
    for(int i=0;i<11;++i)if(h.reserved[i])return -3;
    bool seen[110]={};uint64_t cursor=64+uint64_t(h.groups)*64;
    for(unsigned i=0;i<h.groups;++i) {
        RegionModelGroup g=region_model_group(data,64+i*64);
        if(g.id>=110 || seen[g.id] || !g.channels || g.channels>320 || g.packet_offset!=cursor ||
           g.packet_offset%64 || g.packet_bytes<64 || g.packet_bytes>REGION_MODEL_MAX_PACKET ||
           g.packet_bytes%64 || cursor+g.packet_bytes>bytes)return -4;
        seen[g.id]=true;
        for(int r=0;r<9;++r)if(g.reserved[r])return -4;
        uint32_t p[8];for(unsigned j=0;j<8;++j)p[j]=region_model_word(data,g.packet_offset+j*4);
        if(p[0]!=STREAM_PACKET_MAGIC || p[1]!=STREAM_FP32_PACKET_VERSION || !p[2] || p[2]>6 ||
           p[5]!=g.packet_bytes || p[3]<32+p[2]*48 || p[3]%64 || p[4]%64 ||
           p[4]<p[3] || p[4]>p[5])return -5;
        StreamOp previous={};int expected_q=0;uint64_t weight_end=0;
        for(unsigned j=0;j<p[2];++j) {
            StreamOp op=region_model_op(data,g.packet_offset+32+j*48);
            if(!stream_op_valid(op) || op.weight_offset%64 || op.requant_offset!=expected_q ||
               uint64_t(op.weight_offset)<weight_end || uint64_t(p[4])+op.weight_offset+uint64_t(op.co)*(op.kind?9:op.ci)>p[5] ||
               uint64_t(p[3])+(uint64_t(op.requant_offset)+op.co)*16>p[4])return -6;
            if(j && (previous.co!=op.ci || previous.oh!=op.h || previous.ow!=op.w))return -6;
            if(!j && (g.channels!=unsigned(op.co) || p[6]!=unsigned(op.ci*op.h*op.w)))return -6;
            weight_end=uint64_t(op.weight_offset)+uint64_t(op.co)*(op.kind?9:op.ci);
            for(int c=0;c<op.co;++c) {
                const unsigned char* slot=data+g.packet_offset+p[3]+(op.requant_offset+c)*16;
                if(j+1<p[2]) {
                    StreamRequant q={int32_t(region_model_word(slot,0)),int32_t(region_model_word(slot,4)),
                        int64_t(uint64_t(region_model_word(slot,8))|(uint64_t(region_model_word(slot,12))<<32))};
                    if(q.multiplier<0 || q.multiplier>=8388608 || q.shift<0 || q.shift>40 || q.bias<-(int64_t(1)<<60) || q.bias>(int64_t(1)<<60))return -7;
                } else {
                    StreamOutputAffine affine={region_model_float(slot,0),region_model_float(slot,4),region_model_float(slot,8),region_model_word(slot,12)};
                    if(!stream_output_affine_valid(affine))return -7;
                }
            }
            expected_q+=op.co;previous=op;
        }
        if(p[7]!=unsigned(previous.co*previous.oh*previous.ow))return -6;
        cursor+=g.packet_bytes;
        const uint32_t offsets[]={g.weight_scale_offset,g.bias_offset,g.outgoing_offset};
        for(int k=0;k<3;++k) {
            if(offsets[k]!=cursor || cursor%64 || cursor+uint64_t(g.channels)*4>bytes)return -8;
            for(unsigned c=0;c<g.channels;++c) {
                const uint32_t bits=region_model_word(data,cursor+c*4);const uint32_t magnitude=bits&0x7fffffffu;
                if(k==1) {
                    if(magnitude && (magnitude<0x17800000u || magnitude>0x67800000u))return -9;
                } else if(bits<0x2b800000u || bits>0x53800000u)return -9;
            }
            cursor=(cursor+uint64_t(g.channels)*4+63)/64*64;
        }
    }
    return cursor==bytes?0:-10;
}
