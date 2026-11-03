#pragma once
#include <stdint.h>

// Serialize the at-most-two partial boundary transactions outside the burst.
// Never suppress a real read/write dependence between neighboring channels.
template<class Wide>
static void store_half_128_v2(Wide* results,int address,uint64_t word,bool upper) {
#pragma HLS INLINE off
#pragma HLS LATENCY min=2
    const Wide low_mask=Wide(~uint64_t(0));
    const Wide previous=results[address];
    results[address]=upper ? (previous&low_mask)|(Wide(word)<<64)
                           : (previous&(~low_mask))|Wide(word);
}

template<class Wide>
static void store_final_128(const uint64_t final[16][256],Wide* results,
                            int co,int plane,int channels,int start,int count) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=final complete dim=1
    // Full CHW planes are adjacent across channels; preserve their original order.
    // No partial-word access here. Keep the old guarded writer for every other shape.
    if(start==0 && count==plane && plane>0 && plane<=512 && (plane&3)==0 && channels<co) {
        const int lanes=(co-channels<16)?co-channels:16;
        const int per_lane=plane/4;
        const int total=lanes*per_lane;
        const int base=channels*per_lane;
        int lane=0,within=0;
        for(int linear=0;linear<total;++linear) {
#pragma HLS PIPELINE II=1
            results[base+linear]=Wide(final[lane][2*within])|(Wide(final[lane][2*within+1])<<64);
            if(within==per_lane-1) {within=0;++lane;} else ++within;
        }
    } else {
    const int words=count/2;
    for(int lane=0;lane<16;++lane)if(channels+lane<co) {
        const int begin=((channels+lane)*plane+start)/2;
        const int head=begin&1;
        if(head && words)store_half_128_v2(results,begin/2,final[lane][0],true);
        const int pairs=(words-head)/2;
        for(int i=0;i<pairs;++i) {
#pragma HLS PIPELINE II=1
            const int at=head+i*2;
            results[(begin+head)/2+i]=Wide(final[lane][at])|(Wide(final[lane][at+1])<<64);
        }
        if((words-head)&1)
            store_half_128_v2(results,(begin+words-1)/2,final[lane][words-1],false);
    }
    }
}
