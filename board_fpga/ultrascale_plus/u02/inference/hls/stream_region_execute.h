#pragma once
#include "stream_region_program.h"

// Context::run(group,input,output,scale_scope) returns zero only after the
// complete group output is visible. Caller owns stable metadata and scratch.
// This loop itself performs no transport; its nine group calls are not nine
// hardware submissions when implemented inside one future PL top-level call.
template<class Context> int region_execute(const void* metadata,size_t bytes,
        float* arena,size_t arena_bytes,Context& context) {
    const int status=region_validate(metadata,bytes);
    if(status)return status;
    RegionHeader h;memcpy(&h,metadata,sizeof(h));
#ifndef __SYNTHESIS__
    if(!arena)return -20;
#endif
    if(arena_bytes<h.arena_bytes)return -20;
    const unsigned char* data=static_cast<const unsigned char*>(metadata);
    RegionTensor tensors[REGION_MAX_TENSORS];
    for(unsigned i=0;i<h.tensors;++i)memcpy(&tensors[i],data+h.tensor_offset+i*32,32);
    // Bind group packet geometry before any layout writes or computation.
    for(unsigned i=0;i<h.commands;++i) {
        RegionCommand q;memcpy(&q,data+h.command_offset+i*32,32);
        if(q.opcode==REGION_STREAM && !context.matches(q.group,tensors[q.input0],tensors[q.output0],q.param))return -21;
    }
    for(unsigned i=0;i<h.commands;++i) {
        RegionCommand q;memcpy(&q,data+h.command_offset+i*32,32);
        if(q.opcode==REGION_ALIAS)continue;
        const auto& a=tensors[q.input0];const auto& o=tensors[q.output0];
        const float* input=arena+a.offset/4;float* output=arena+o.offset/4;
        if(q.opcode==REGION_CONCAT) {
            const auto& b=tensors[q.input1];const float* second=arena+b.offset/4;
            for(unsigned j=0;j<a.bytes/4;++j)output[j]=input[j];
            for(unsigned j=0;j<b.bytes/4;++j)output[a.bytes/4+j]=second[j];
        } else if(q.opcode==REGION_SHUFFLE) {
            const unsigned half=a.c/2,plane=a.h*a.w;
            for(unsigned c=0;c<half;++c)for(unsigned p=0;p<plane;++p) {
                output[(2*c)*plane+p]=input[c*plane+p];
                output[(2*c+1)*plane+p]=input[(c+half)*plane+p];
            }
        } else if(q.opcode==REGION_STREAM) {
            const int rc=context.run(q.group,input,output,q.param);
            if(rc)return rc;
        } else return -22;
    }
    return 0;
}
