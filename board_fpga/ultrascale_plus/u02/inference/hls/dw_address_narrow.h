#pragma once
#include <ap_int.h>

// Called only after stream_op_valid, the 768-word footprint check, and the
// signed x/y bounds checks. Maximum h=64,w=48,y=63,x=47,channel tile=304.
// MACs, packing and the public packet/register ABI do not change.
struct DwReadAddress {
    ap_uint<15> word;
    ap_uint<3> byte;
};
static DwReadAddress dw_read_address_narrow(int channels,int y,int x,int h,int w) {
#pragma HLS INLINE
    const ap_uint<6> row=y,col=x,width=w;
    const ap_uint<7> height=h; // 64 must remain representable.
    const ap_uint<13> plane=height*width;
    const ap_uint<10> pitch=(plane+ap_uint<13>(7))>>3;
    const ap_uint<12> pixel=ap_uint<12>(row*width)+col;
    const ap_uint<5> block=ap_uint<9>(channels)>>4;
    DwReadAddress result;
    // Uncorrelated maxima: 19*384+383=7679, well within fifteen bits.
    // The separate validated feature footprint bounds actual accesses <768.
    result.word=ap_uint<15>(block*pitch)+ap_uint<15>(pixel>>3);
    result.byte=pixel.range(2,0);
    return result;
}
