#pragma once
#include <stdint.h>

// Bit-preserving CHW INT32 writer. Wide is ap_uint<128> in HLS and native
// unsigned __int128 in the independent packing test. No arithmetic on sums.
// Full words never read results. A boundary word preserves its untouched half.
template<class Wide>
static void store_final_128(const uint64_t final[16][256],Wide* results,
                            int co,int plane,int channels,int start,int count) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=final complete dim=1
    const int words=count/2;
    const Wide low_mask=Wide(~uint64_t(0));
    for(int lane=0;lane<16;++lane)if(channels+lane<co) {
        const int begin=((channels+lane)*plane+start)/2;
        const int end=begin+words;
        if((begin%2)==0 && (words%2)==0) {
            for(int i=0;i<words/2;++i) {
#pragma HLS PIPELINE II=1
                results[begin/2+i]=Wide(final[lane][i*2])|(Wide(final[lane][i*2+1])<<64);
            }
        } else {
        for(int address=begin/2;address<(end+1)/2;++address) {
#pragma HLS PIPELINE II=1
            const int lo=address*2-begin,hi=lo+1;
            const bool valid_lo=lo>=0 && lo<words,valid_hi=hi>=0 && hi<words;
            Wide value;
            if(valid_lo && valid_hi) {
                value=Wide(final[lane][lo])|(Wide(final[lane][hi])<<64);
            } else {
                value=results[address];
                if(valid_lo)value=(value&(~low_mask))|Wide(final[lane][lo]);
                if(valid_hi)value=(value&low_mask)|(Wide(final[lane][hi])<<64);
            }
            results[address]=value;
        }
        }
    }
}
