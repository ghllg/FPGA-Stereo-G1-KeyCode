#include "quantize_stream.h"

extern "C" int pose_quantize_stream(uint64_t* b,int count,int source_word,int output_word) {
#ifndef POSE_QUANT_SUBFUNCTION
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=features depth=1048576 max_read_burst_length=64
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=count bundle=control
#pragma HLS INTERFACE s_axilite port=source_word bundle=control
#pragma HLS INTERFACE s_axilite port=output_word bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif
#pragma HLS INLINE off
    if (count<1 || count>1048576 || source_word<64 ||
        source_word>1048576-(count+1)/2 || output_word<source_word+(count+1)/2 ||
        output_word>1048576-(count+7)/8) return -1;
    uint32_t thresholds[7][128];
#pragma HLS ARRAY_PARTITION variable=thresholds complete dim=1
#pragma HLS RESOURCE variable=thresholds core=RAM_2P_BRAM
    for (int i=0;i<64;++i) {
#pragma HLS PIPELINE II=1
        const uint64_t word=b[i];
        for (int stage=0;stage<7;++stage) {
#pragma HLS UNROLL
            thresholds[stage][i*2]=uint32_t(word);
            thresholds[stage][i*2+1]=uint32_t(word>>32);
        }
    }
    uint64_t packed=0;
    const int words=(count+1)/2;
    for (int w=0;w<words;++w) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=b inter false
        // Input and output ranges do not overlap (checked above). Each output
        // word is written exactly once, so there are no inter-iteration hazards.
        const uint64_t word=b[source_word+w];
        uint8_t result[2];
#pragma HLS ARRAY_PARTITION variable=result complete
        for (int lane=0;lane<2;++lane) {
#pragma HLS UNROLL
            const uint32_t value=uint32_t(word>>(lane*32));
            const uint32_t magnitude=value&0x7fffffffu;
            unsigned index=0;
            for (int stage=0;stage<7;++stage) {
#pragma HLS UNROLL
                const unsigned candidate=index|(1u<<(6-stage));
                if (magnitude>=thresholds[stage][candidate-1]) index=candidate;
            }
            result[lane]=w*2+lane<count ? uint8_t(value>>31 ? -int(index) : int(index)) : 0;
        }
        if ((w&3)==0) packed=0;
        packed|=uint64_t(result[0])<<((w&3)*16);
        packed|=uint64_t(result[1])<<((w&3)*16+8);
        if ((w&3)==3 || w==words-1) b[output_word+w/4]=packed;
    }
    return 0;
}
