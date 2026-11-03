#pragma once
#include <ap_int.h>

struct DwReadAddress {
    ap_uint<15> word;
    ap_uint<3> byte;
};
static DwReadAddress dw_read_address_narrow(int channels,int y,int x,int h,int w) {
#pragma HLS INLINE
    const ap_uint<6> row=y,col=x,width=w;
    const ap_uint<7> height=h;
    const ap_uint<13> plane=height*width;
    const ap_uint<10> pitch=(plane+ap_uint<13>(7))>>3;
    // Keep the six-bit row truncation before multiplication. HLS2018.3's
    // inferred DSP preadder in the first candidate absorbed the row addition
    // and lost its modulo64 behavior at top padding (63+1 must become0).
    // UG902 2018.3 documents Mul_LUT for native C POD types, so bind the
    // native unsigned product rather than applying it to an ap_uint object.
    const unsigned row_product=row.to_uint()*width.to_uint();
#pragma HLS RESOURCE variable=row_product core=Mul_LUT
    const ap_uint<12> pixel=ap_uint<12>(row_product)+col;
    const ap_uint<5> block=ap_uint<9>(channels)>>4;
    DwReadAddress result;
    result.word=ap_uint<15>(block*pitch)+ap_uint<15>(pixel>>3);
    result.byte=pixel.range(2,0);
    return result;
}
