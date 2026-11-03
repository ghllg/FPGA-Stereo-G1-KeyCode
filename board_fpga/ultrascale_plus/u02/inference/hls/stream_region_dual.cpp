#include "region_dual_compute.h"
#include "region_model_words.h"
#include "region_boundary_words.h"
#include "stream_region_program.h"

struct RegionDualWorkspace {
    uint64_t packet[2048],quantized[12288];
    float weight[320],bias[320],outgoing[320];
};
// Both modes converge at this single call site and share the same MAC engine.
static int region_dual_group(const uint64_t* commands,const uint64_t* model,uint64_t* arena,
        RegionDualWorkspace& work,bool legacy,unsigned group,size_t input_offset,size_t output_offset,
        unsigned scope,int operators,int packet_bytes) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION instances=region_dual_compute limit=1 function
#pragma HLS RESOURCE variable=work.quantized core=RAM_2P_URAM
#pragma HLS RESOURCE variable=work.packet core=RAM_2P_BRAM
    if(!legacy) {
        const unsigned groups=region_model_words_word(model,12);
        RegionModelGroup selected={};bool found=false;
        for(unsigned i=0;i<groups;++i) {
            RegionModelGroup g=region_model_words_group(model,64+i*64);
            if(g.id==group){selected=g;found=true;break;}
        }
        if(!found)return -30;
        for(unsigned i=0;i<selected.packet_bytes/8;++i)work.packet[i]=model[selected.packet_offset/8+i];
        if(scope!=uint32_t(work.packet[4]))return -31;
        for(unsigned i=0;i<selected.channels;++i) {
            work.weight[i]=region_model_words_float(model,selected.weight_scale_offset+i*4);
            work.bias[i]=region_model_words_float(model,selected.bias_offset+i*4);
            work.outgoing[i]=region_model_words_float(model,selected.outgoing_offset+i*4);
        }
        const unsigned elements=uint32_t(work.packet[3]);
        const int status=region_prepare_boundary_words_to(arena,input_offset,elements,work.weight,work.bias,work.outgoing,
            selected.channels,work.packet,selected.packet_bytes,RegionWordDestination{work.quantized},sizeof(work.quantized));
        if(status)return status;
        operators=uint32_t(work.packet[1]);packet_bytes=selected.packet_bytes;
    }
    return region_dual_compute(commands,model,work.packet,work.quantized,arena+output_offset/2,
        operators,packet_bytes,STREAM_PACKET_MAGIC,legacy);
}

extern "C" int u02_region_dual(const uint64_t* metadata,const uint64_t* model,uint64_t* arena,
        int metadata_bytes,int model_bytes,int arena_bytes) {
#pragma HLS ALLOCATION instances=region_dual_group limit=1 function
#pragma HLS INTERFACE m_axi port=metadata offset=slave bundle=commands depth=131072 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=model offset=slave bundle=weights depth=131072 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=arena offset=slave bundle=features depth=262144 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=metadata bundle=control
#pragma HLS INTERFACE s_axilite port=model bundle=control
#pragma HLS INTERFACE s_axilite port=arena bundle=control
#pragma HLS INTERFACE s_axilite port=metadata_bytes bundle=control
#pragma HLS INTERFACE s_axilite port=model_bytes bundle=control
#pragma HLS INTERFACE s_axilite port=arena_bytes bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    RegionDualWorkspace work;
    const bool legacy=arena_bytes==int(STREAM_PACKET_MAGIC);
    if(legacy && (metadata_bytes<1 || metadata_bytes>6 || model_bytes<64 || model_bytes>int(STREAM_MAX_PACKET_BYTES) || model_bytes%64))return -1;
    if(!legacy && (metadata_bytes<64 || metadata_bytes>16384 || metadata_bytes%64 || model_bytes<128 || model_bytes>1048576 ||
        model_bytes%64 || arena_bytes<64 || arena_bytes>int(REGION_MAX_ARENA)))return -40;
    // Metadata/tensor snapshots are separate from AXI word ports. A single
    // loop below invokes the shared group function for either protocol.
    unsigned char program_copy[16384];
    RegionHeader program={};RegionTensor tensors[REGION_MAX_TENSORS];
    if(!legacy) {
        for(unsigned i=0;i<unsigned(metadata_bytes)/8;++i) {
            const uint64_t word=metadata[i];
            for(unsigned j=0;j<8;++j)program_copy[i*8+j]=uint8_t(word>>(j*8));
        }
        int status=region_validate(program_copy,metadata_bytes);if(status)return status;
        status=region_model_words_validate(model,model_bytes);if(status)return status;
        memcpy(&program,program_copy,64);
    if(unsigned(arena_bytes)<program.arena_bytes)return -20;
    for(unsigned i=0;i<program.tensors;++i)memcpy(&tensors[i],program_copy+program.tensor_offset+i*32,32);
    // Pre-bind all shapes and require aligned full-word convolution outputs.
    for(unsigned i=0;i<program.commands;++i) {
        RegionCommand q;memcpy(&q,program_copy+program.command_offset+i*32,32);
        if(q.opcode!=REGION_STREAM)continue;
        bool found=false;
        for(unsigned j=0;j<region_model_words_word(model,12);++j) {
            RegionModelGroup g=region_model_words_group(model,64+j*64);if(g.id!=q.group)continue;
            StreamOp first=region_model_words_op(model,g.packet_offset+32);
            unsigned count=region_model_words_word(model,g.packet_offset+8);
            StreamOp last=region_model_words_op(model,g.packet_offset+32+(count-1)*48);
            const RegionTensor a=tensors[q.input0],o=tensors[q.output0];
            if(q.param!=unsigned(first.kind) || a.c!=unsigned(first.ci) || a.h!=unsigned(first.h) || a.w!=unsigned(first.w) ||
                o.c!=unsigned(last.co) || o.h!=unsigned(last.oh) || o.w!=unsigned(last.ow) || o.offset%8)return -21;
            found=true;break;
        }
        if(!found)return -21;
    }
    }
    for(unsigned i=0;i<(legacy?1:program.commands);++i) {
        RegionCommand q={};RegionTensor a={},o={};
        if(legacy)q.opcode=REGION_STREAM;
        else {memcpy(&q,program_copy+program.command_offset+i*32,32);a=tensors[q.input0];o=tensors[q.output0];}
        if(q.opcode==REGION_ALIAS)continue;
        if(q.opcode==REGION_CONCAT) {
            const RegionTensor b=tensors[q.input1];
            for(unsigned j=0;j<a.bytes/4;++j)region_word_store(arena,o.offset/4+j,region_word_float(arena,a.offset/4+j));
            for(unsigned j=0;j<b.bytes/4;++j)region_word_store(arena,(o.offset+a.bytes)/4+j,region_word_float(arena,b.offset/4+j));
        } else if(q.opcode==REGION_SHUFFLE) {
            unsigned half=a.c/2,plane=a.h*a.w;
            for(unsigned c=0;c<half;++c)for(unsigned p=0;p<plane;++p) {
                region_word_store(arena,o.offset/4+2*c*plane+p,region_word_float(arena,a.offset/4+c*plane+p));
                region_word_store(arena,o.offset/4+(2*c+1)*plane+p,region_word_float(arena,a.offset/4+(c+half)*plane+p));
            }
        } else if(q.opcode==REGION_STREAM) {
            const int status=region_dual_group(metadata,model,arena,work,legacy,q.group,a.offset/4,o.offset/4,q.param,
                metadata_bytes,model_bytes);
            if(status)return status;
        } else return -22;
    }
    return 0;
}
