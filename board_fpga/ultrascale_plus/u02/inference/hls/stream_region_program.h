#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Separate metadata ABI for the region dispatcher. No existing stream packet
// or device ioctl accepts it. Addresses are offsets in an owned scratch arena.
static const uint32_t REGION_MAGIC=0x52323055u, REGION_VERSION=1;
static const uint32_t REGION_NONE=0xffffffffu;
static const uint32_t REGION_MAX_TENSORS=128, REGION_MAX_COMMANDS=64;
static const uint32_t REGION_MAX_ARENA=2*1024*1024;
enum RegionOpcode {REGION_ALIAS=1, REGION_CONCAT=2, REGION_SHUFFLE=3, REGION_STREAM=4};
struct RegionHeader {
    uint32_t magic,version,bytes,tensors,commands,arena_bytes,tensor_offset,command_offset;
    uint32_t stream_commands,entry_node,reserved[6];
};
struct RegionTensor {uint32_t offset,bytes,c,h,w,root,flags,reserved;};
struct RegionCommand {uint32_t opcode,input0,input1,output0,output1,group,param,reserved;};
static_assert(sizeof(RegionHeader)==64,"Region header ABI");
static_assert(sizeof(RegionTensor)==32,"Region tensor ABI");
static_assert(sizeof(RegionCommand)==32,"Region command ABI");

inline bool region_overlap(const RegionTensor& a,const RegionTensor& b) {
    return uint64_t(a.offset)<uint64_t(b.offset)+b.bytes && uint64_t(b.offset)<uint64_t(a.offset)+a.bytes;
}

// Validates a private, stable metadata snapshot. A future kernel must copy
// userspace metadata before validation and retain DMA ownership until idle.
// Packet/model identity and group arithmetic need their own validation later.
inline int region_validate(const void* raw,size_t bytes) {
#ifndef __SYNTHESIS__
    if(!raw)return -1;
#endif
    if(bytes<sizeof(RegionHeader) || bytes>16384 || bytes%64)return -1;
    const unsigned char* data=static_cast<const unsigned char*>(raw);
    RegionHeader h;memcpy(&h,data,sizeof(h));
    if(h.magic!=REGION_MAGIC || h.version!=REGION_VERSION || h.bytes!=bytes ||
       !h.tensors || h.tensors>REGION_MAX_TENSORS || !h.commands || h.commands>REGION_MAX_COMMANDS ||
       !h.arena_bytes || h.arena_bytes>REGION_MAX_ARENA || h.arena_bytes%64 ||
       !h.stream_commands || h.stream_commands>h.commands || h.tensor_offset!=64 ||
       h.command_offset%64 || uint64_t(h.tensor_offset)+uint64_t(h.tensors)*32>h.command_offset ||
       uint64_t(h.command_offset)+uint64_t(h.commands)*32>bytes)return -2;
    for(int i=0;i<6;++i)if(h.reserved[i])return -3;
    RegionTensor tensors[REGION_MAX_TENSORS];bool ready[REGION_MAX_TENSORS]={};
    unsigned ingress=0,egress=0;
    for(unsigned i=0;i<h.tensors;++i) {
        memcpy(&tensors[i],data+h.tensor_offset+i*32,32);
        const auto& t=tensors[i];
        if(!t.c || t.c>320 || !t.h || t.h>64 || !t.w || t.w>48 || t.offset%4 ||
           uint64_t(t.c)*t.h*t.w>98304 || uint64_t(t.c)*t.h*t.w*4!=t.bytes ||
           uint64_t(t.offset)+t.bytes>h.arena_bytes || t.root>=h.tensors || t.flags>3 || t.reserved)return -4;
        if(t.flags&1){ready[i]=true;++ingress;}
        if(t.flags&2)++egress;
    }
    if(!ingress || !egress)return -5;
    for(unsigned i=0;i<h.tensors;++i) {
        const auto& t=tensors[i];const auto& root=tensors[t.root];
        if(root.root!=t.root || t.offset<root.offset || uint64_t(t.offset)+t.bytes>uint64_t(root.offset)+root.bytes)return -6;
        if((t.flags&1) && t.root!=i)return -6;
        if(ready[i])for(unsigned j=0;j<i;++j)if(ready[j] && region_overlap(t,tensors[j]))return -6;
    }
    bool groups[110]={};unsigned streams=0;
    for(unsigned i=0;i<h.commands;++i) {
        RegionCommand q;memcpy(&q,data+h.command_offset+i*32,32);
        if(q.reserved || q.input0>=h.tensors || !ready[q.input0] || q.output0>=h.tensors || ready[q.output0])return -7;
        const auto& a=tensors[q.input0];const auto& o=tensors[q.output0];
        if(q.opcode==REGION_ALIAS) {
            if(q.input1!=REGION_NONE || q.group!=REGION_NONE || q.param>1 || q.output1>=h.tensors ||
               q.output1==q.output0 || ready[q.output1])return -8;
            const auto& p=tensors[q.output1];
            if(o.root!=a.root || p.root!=a.root || o.h!=a.h || p.h!=a.h || o.w!=a.w || p.w!=a.w)return -8;
            if(q.param==0) {
                if(o.offset!=a.offset || p.offset!=a.offset || o.bytes!=a.bytes || p.bytes!=a.bytes)return -8;
            } else {
                if(a.c%2 || o.c!=a.c/2 || p.c!=a.c/2 || o.offset!=a.offset ||
                   uint64_t(p.offset)!=uint64_t(a.offset)+o.bytes || uint64_t(o.bytes)+p.bytes!=a.bytes)return -8;
            }
            ready[q.output1]=true;
        } else {
            if(q.output1!=REGION_NONE || o.root!=q.output0)return -9;
            // Conservative first implementation has no storage reuse: reject
            // writes over any earlier result or aliased live branch.
            for(unsigned j=0;j<h.tensors;++j)if(ready[j] && region_overlap(o,tensors[j]))return -9;
            if(q.opcode==REGION_CONCAT) {
                if(q.input1>=h.tensors || !ready[q.input1] || q.group!=REGION_NONE || q.param)return -10;
                const auto& b=tensors[q.input1];
                if(a.h!=b.h || a.w!=b.w || o.h!=a.h || o.w!=a.w || uint64_t(a.c)+b.c!=o.c)return -10;
            } else if(q.opcode==REGION_SHUFFLE) {
                if(q.input1!=REGION_NONE || q.group!=REGION_NONE || q.param!=2 || a.c%2 ||
                   o.c!=a.c || o.h!=a.h || o.w!=a.w)return -11;
            } else if(q.opcode==REGION_STREAM) {
                if(q.input1!=REGION_NONE || q.group>=110 || groups[q.group] || q.param>1)return -12;
                groups[q.group]=true;++streams;
            } else return -13;
        }
        ready[q.output0]=true;
    }
    if(streams!=h.stream_commands)return -14;
    for(unsigned i=0;i<h.tensors;++i)if(!ready[i])return -15;
    return 0;
}
