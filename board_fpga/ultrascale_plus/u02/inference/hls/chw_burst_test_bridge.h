#pragma once
#include <ap_int.h>

extern "C" int u02_stream_chw_burst(const uint64_t*,const uint64_t*,ap_uint<128>*,int,int,int);
// Test-only adapter: preserve the old complete-network checker and its guards.
// All its callers allocate TEST_OUTPUT_WORDS+8 uint64_t entries. This is never
// deployed as a Host or included in the synthesized kernel.
static int u02_stream_128_bridge(const uint64_t* a,const uint64_t* b,uint64_t* c,int m,int n,int k) {
    std::vector<ap_uint<128> > wide((TEST_OUTPUT_WORDS+8)/2);
    for(size_t i=0;i<wide.size();++i)wide[i]=ap_uint<128>(c[2*i])|(ap_uint<128>(c[2*i+1])<<64);
    const int result=u02_stream_chw_burst(a,b,wide.data(),m,n,k);
    for(size_t i=0;i<wide.size();++i) {
        c[2*i]=uint64_t(wide[i].range(63,0));c[2*i+1]=uint64_t(wide[i].range(127,64));
    }
    return result;
}
#define u02_stream u02_stream_128_bridge
