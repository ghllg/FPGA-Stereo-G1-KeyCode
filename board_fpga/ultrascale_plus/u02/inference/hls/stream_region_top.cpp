#include "stream_region_hls_context.h"

extern "C" int u02_region(const unsigned char* metadata,const unsigned char* model,
        float* arena,int metadata_bytes,int model_bytes,int arena_bytes) {
#pragma HLS INTERFACE m_axi port=metadata offset=slave bundle=commands depth=16384
#pragma HLS INTERFACE m_axi port=model offset=slave bundle=weights depth=1048576
#pragma HLS INTERFACE m_axi port=arena offset=slave bundle=features depth=524288 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=metadata bundle=control
#pragma HLS INTERFACE s_axilite port=model bundle=control
#pragma HLS INTERFACE s_axilite port=arena bundle=control
#pragma HLS INTERFACE s_axilite port=metadata_bytes bundle=control
#pragma HLS INTERFACE s_axilite port=model_bytes bundle=control
#pragma HLS INTERFACE s_axilite port=arena_bytes bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if(metadata_bytes<64 || metadata_bytes>16384 || model_bytes<128 || model_bytes>1048576 ||
       arena_bytes<64 || arena_bytes>int(REGION_MAX_ARENA))return -40;
    unsigned char private_metadata[16384];
    memcpy(private_metadata,metadata,metadata_bytes);
    int status=region_validate(private_metadata,metadata_bytes);if(status)return status;
    status=region_model_validate(model,model_bytes);if(status)return status;
    RegionHlsContext context(region_model_word(model,12));
    RegionHeader program;memcpy(&program,private_metadata,64);
    if(unsigned(arena_bytes)<program.arena_bytes)return -20;
    RegionTensor tensors[REGION_MAX_TENSORS];
    for(unsigned i=0;i<program.tensors;++i)memcpy(&tensors[i],private_metadata+program.tensor_offset+i*32,32);
    // Explicit AXI model argument: 2018.3 cannot wire a bus pointer captured
    // as a class member. Bind all stream geometry before any scratch writes.
    for(unsigned i=0;i<program.commands;++i) {
        RegionCommand q;memcpy(&q,private_metadata+program.command_offset+i*32,32);
        if(q.opcode==REGION_STREAM && !context.matches(model,q.group,tensors[q.input0],tensors[q.output0],q.param))return -21;
    }
    for(unsigned i=0;i<program.commands;++i) {
        RegionCommand q;memcpy(&q,private_metadata+program.command_offset+i*32,32);
        if(q.opcode==REGION_ALIAS)continue;
        const RegionTensor a=tensors[q.input0],o=tensors[q.output0];
        const float* input=arena+a.offset/4;float* output=arena+o.offset/4;
        if(q.opcode==REGION_CONCAT) {
            const RegionTensor b=tensors[q.input1];
            for(unsigned j=0;j<a.bytes/4;++j)output[j]=input[j];
            for(unsigned j=0;j<b.bytes/4;++j)output[a.bytes/4+j]=arena[b.offset/4+j];
        } else if(q.opcode==REGION_SHUFFLE) {
            const unsigned half=a.c/2,plane=a.h*a.w;
            for(unsigned c=0;c<half;++c)for(unsigned p=0;p<plane;++p) {
                output[2*c*plane+p]=input[c*plane+p];
                output[(2*c+1)*plane+p]=input[(c+half)*plane+p];
            }
        } else if(q.opcode==REGION_STREAM) {
            status=context.run(model,q.group,input,output,q.param);if(status)return status;
        } else return -22;
    }
    return 0;
}
