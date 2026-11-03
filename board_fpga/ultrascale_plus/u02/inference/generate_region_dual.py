"""Derive packed-word dual-mode arithmetic from verified scalar/packet sources."""
import argparse
import json
import re
from pathlib import Path
from lower_stream_region import U02, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    hls=U02/'inference/hls'
    packet=hls/'stream_overlap_fp32_output.cpp'
    model=hls/'stream_region_model_packet.h'
    boundary=hls/'stream_region_boundary.h'
    source=packet.read_text()
    compute='\n'.join(line for line in source.splitlines() if not line.startswith('#pragma HLS INTERFACE'))+'\n'
    signature='extern "C" int u02_stream(const uint64_t* a,const uint64_t* b,uint64_t* c,int m,int n,int k)'
    if compute.count(signature)!=1:raise ValueError('Compute signature changed')
    compute=compute.replace(signature,'static int region_dual_compute(const uint64_t* commands,const uint64_t* model,\n'
        '        const uint64_t* patched_packet,const uint64_t* quantized,uint64_t* c,int m,int n,int k,bool legacy)')
    compute=re.sub(r'\ba\[([^\]]+)\]',r'(legacy?commands[\1]:patched_packet[\1])',compute)
    compute=re.sub(r'\bb\[([^\]]+)\]',r'(legacy?model[\1]:quantized[\1])',compute)
    if re.search(r'\b[ab]\[',compute):raise ValueError('Unconverted AXI read')
    (args.output_dir/'region_dual_compute.h').write_text('// Generated; original packed FP32 output and one compute engine.\n'+compute)

    raw=model.read_text();start=raw.index('inline uint32_t region_model_word')
    words=raw[start:]
    words=re.sub(r'inline uint32_t region_model_word.*?(?=inline float region_model_float)',
        'inline uint32_t region_model_word(const uint64_t* data,size_t at) {\n'
        '    return uint32_t(data[at/8]>>((at%8)*8));\n}\n',words,flags=re.S)
    words=words.replace('const unsigned char*','const uint64_t*')
    words=words.replace('const void* raw','const uint64_t* raw')
    words=words.replace('static_cast<const uint64_t*>(raw)','raw')
    words=words.replace('const uint64_t* slot=data+g.packet_offset+p[3]+(op.requant_offset+c)*16;',
        'const size_t slot=g.packet_offset+p[3]+(op.requant_offset+c)*16;')
    words=re.sub(r'region_model_(word|float)\(slot,(\d+)\)',r'region_model_\1(data,slot+\2)',words)
    words=words.replace('region_model_','region_model_words_')
    if 'slot=data+' in words:raise ValueError('Word model slot translation incomplete')
    (args.output_dir/'region_model_words.h').write_text('#pragma once\n#include "stream_region_model_packet.h"\n'+words)

    raw=boundary.read_text();start=raw.index('template<class Destination>')
    end=raw.index('inline int region_prepare_boundary(',start)
    arithmetic=raw[start:end].replace('region_prepare_boundary_to','region_prepare_boundary_words_to')
    arithmetic=arithmetic.replace('const float* input,size_t elements','const uint64_t* input,size_t input_offset,size_t elements')
    arithmetic=re.sub(r'input\[([^\]]+)\]',r'region_word_float(input,input_offset+(\1))',arithmetic)
    prefix='#pragma once\n#include "stream_region_boundary.h"\n'
    prefix+='inline float region_word_float(const uint64_t* data,size_t at) {\n'
    prefix+='    union Bits {uint32_t bits;float value;};Bits result;result.bits=uint32_t(data[at/2]>>((at%2)*32));return result.value;\n}\n'
    prefix+='inline void region_word_store(uint64_t* data,size_t at,float value) {\n'
    prefix+='    union Bits {uint32_t bits;float value;};Bits packed;packed.value=value;const unsigned shift=unsigned(at%2)*32;\n'
    prefix+='    data[at/2]=(data[at/2]&~(uint64_t(0xffffffffu)<<shift))|(uint64_t(packed.bits)<<shift);\n}\n'
    (args.output_dir/'region_boundary_words.h').write_text(prefix+arithmetic)
    state=dict(source_sha256={str(p):sha(p) for p in (packet,model,boundary,Path(__file__))},
        files_sha256={p.name:sha(p) for p in args.output_dir.glob('*.h')},
        hardware_verified=False,scope='Packed64 readers; unchanged packet2 arithmetic and dynamic boundary math; no deployed backend')
    (args.output_dir/'generation.json').write_text(json.dumps(state,indent=2)+'\n')
    print('U02_REGION_DUAL_GENERATED',args.output_dir)


if __name__=='__main__':main()
