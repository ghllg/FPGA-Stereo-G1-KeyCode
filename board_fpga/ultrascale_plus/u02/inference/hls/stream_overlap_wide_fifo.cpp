// Keep accumulation rounds sequential; only distinct pixels pipeline.
// Cache each operator's requant coefficients in a single contiguous AXI read.
// Generalized from V13A: one 16x16 PW array, 16 DW lanes, four affine lanes.
// Only channel parallelism changes; packet ABI and quantization stay identical.
// One command executes a whole graph segment; intermediates remain on chip.
#include "stream_packet.h"
#include <ap_int.h>
#include <hls_stream.h>

typedef ap_uint<512> Sums;
static uint32_t word32(uint64_t value,int upper) {return uint32_t(value>>(upper*32));}
static int8_t feature(const uint64_t data[16][768],int bank,int channel,int pixel,int plane) {
#pragma HLS INLINE
    const int pitch=(plane+7)/8;
    const uint64_t word=data[channel%16][channel/16*pitch+pixel/8];
    return int8_t(word>>((pixel%8)*8));
}
// The output-channel tile always starts at a multiple of sixteen. Read each
// physical bank explicitly so the unrolled DW lanes cannot alias one port.
static int8_t dw_feature(const uint64_t data[16][768],int bank,int channels,int lane,int pixel,int plane) {
#pragma HLS INLINE
    const int at=(channels/16)*((plane+7)/8)+pixel/8;
    const uint64_t word=data[lane][at];
    return int8_t(word>>((pixel%8)*8));
}
static void unpack_weights(const uint64_t words[12800],int8_t weights[16][320],
                           int channels,int outputs,int kernel) {
#pragma HLS INLINE off
    const int word_count=(outputs*kernel+7)/8,blocks=(kernel+15)/16;
    for(int lane=0;lane<16;++lane) {
#pragma HLS LOOP_FLATTEN off
        for(int block=0;block<blocks;++block) {
#pragma HLS PIPELINE II=1
            const int offset=(channels+lane)*kernel+block*16,at=offset/8,shift=(offset%8)*8;
            const uint64_t a=at<word_count?words[at]:0;
            const uint64_t b=at+1<word_count?words[at+1]:0;
            const uint64_t c=shift && at+2<word_count?words[at+2]:0;
            const uint64_t lo=(a>>shift)|(shift?b<<(64-shift):0);
            const uint64_t hi=(b>>shift)|(shift?c<<(64-shift):0);
            for(int j=0;j<16;++j) {
#pragma HLS UNROLL
                const uint64_t packed=j<8?lo:hi;
                weights[lane][block*16+j]=(channels+lane<outputs && block*16+j<kernel)
                    ?int8_t(packed>>((j%8)*8)):int8_t(0);
            }
        }
    }
}
static int8_t requant(int32_t sum,const StreamRequant& q,bool relu) {
#pragma HLS INLINE
    const ap_int<32> value=sum;const ap_int<24> multiplier=q.multiplier;
    const ap_int<56> product=value*multiplier;
#pragma HLS RESOURCE variable=product core=MulnS latency=3
    const int64_t biased=int64_t(product)+q.bias;
    const uint64_t magnitude=biased<0?uint64_t(-biased):uint64_t(biased);
    const uint64_t rounded=(magnitude+(q.shift?(uint64_t(1)<<(q.shift-1)):0))>>q.shift;
    const int64_t v=biased<0?-int64_t(rounded):int64_t(rounded);
    return int8_t(v>127?127:(v<(relu?0:-127)?(relu?0:-127):v));
}
static void pw_mac(const uint64_t data[16][768],const int8_t weights[16][320],
                   StreamOp op,int bank,int start,int count,hls::stream<Sums>& output) {
#pragma HLS INLINE off
    int32_t partial[16][128];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
#pragma HLS RESOURCE variable=partial core=RAM_2P_LUTRAM
    const int blocks=(op.ci+15)/16;
    for(int block=0;block<blocks;++block) {
#pragma HLS LOOP_FLATTEN off
      for(int pixel=0;pixel<count;++pixel) {
#pragma HLS PIPELINE II=1
        int8_t inputs[16];
#pragma HLS ARRAY_PARTITION variable=inputs complete
        for(int k=0;k<16;++k) {
#pragma HLS UNROLL
            const int ci=block*16+k;
            inputs[k]=ci<op.ci?feature(data,bank,ci,start+pixel,op.h*op.w):int8_t(0);
        }
        Sums packed=0;
        for(int lane=0;lane<16;++lane) {
#pragma HLS UNROLL
            int32_t sums[16];
#pragma HLS ARRAY_PARTITION variable=sums complete
            for(int k=0;k<16;++k) {
#pragma HLS UNROLL
                const int16_t product=int16_t(inputs[k])*int16_t(weights[lane][block*16+k]);
#pragma HLS RESOURCE variable=product core=MulnS latency=2
                sums[k]=product;
            }
            for(int width=8;width>0;width/=2) {
#pragma HLS UNROLL
                for(int k=0;k<width;++k) {
#pragma HLS UNROLL
                    sums[k]+=sums[k+width];
                }
            }
            const int32_t total=(block?partial[lane][pixel]:0)+sums[0];
            partial[lane][pixel]=total;packed.range(lane*32+31,lane*32)=uint32_t(total);
        }
        if(block==blocks-1)output.write(packed);
      }
    }
}
static void dw_mac(const uint64_t data[16][768],const int8_t weights[16][320],
                   StreamOp op,int bank,int channels,int start,int count,hls::stream<Sums>& output) {
#pragma HLS INLINE off
    int32_t partial[16][128];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
#pragma HLS RESOURCE variable=partial core=RAM_2P_LUTRAM
    const int initial_y=start/op.ow,initial_x=start%op.ow;
    for(int kernel=0;kernel<9;++kernel) {
#pragma HLS LOOP_FLATTEN off
      int oy=initial_y,ox=initial_x;
      for(int pixel=0;pixel<count;++pixel) {
#pragma HLS PIPELINE II=1
        const int y=oy*op.stride-op.pad+kernel/3,x=ox*op.stride-op.pad+kernel%3;
        Sums packed=0;
        for(int lane=0;lane<16;++lane) {
#pragma HLS UNROLL
            const int ch=channels+lane;
            const int8_t input=(ch<op.ci && y>=0 && y<op.h && x>=0 && x<op.w)?dw_feature(data,bank,channels,lane,y*op.w+x,op.h*op.w):int8_t(0);
            const int16_t product=int16_t(input)*int16_t(weights[lane][kernel]);
#pragma HLS RESOURCE variable=product core=MulnS latency=2
            const int32_t total=(kernel?partial[lane][pixel]:0)+product;
            partial[lane][pixel]=total;packed.range(lane*32+31,lane*32)=uint32_t(total);
        }
        if(kernel==8) output.write(packed);
        if(ox==op.ow-1){ox=0;++oy;}else ++ox;
      }
    }
}
static void mac_task(const uint64_t data[16][768],const int8_t weights[16][320],StreamOp op,
                     int bank,int channels,int start,int count,hls::stream<Sums>& sums) {
#pragma HLS INLINE off
    if(op.kind)dw_mac(data,weights,op,bank,channels,start,count,sums);
    else pw_mac(data,weights,op,bank,start,count,sums);
}

// One shared MAC/requant/store pipeline serves final and intermediate tiles.
// Intermediate words stream into the other feature bank. Final sums collect
// into a small tile buffer so DDR still receives contiguous per-channel bursts.
typedef ap_uint<1024> FeatureWords;
static void requant_words(hls::stream<Sums>& input,hls::stream<FeatureWords>& output,
                          const StreamRequant coefficients[16],StreamOp op,int count,bool last) {
#pragma HLS INLINE off
    uint64_t words[16];
#pragma HLS ARRAY_PARTITION variable=words complete
    Sums values=0;
    for(int step=0;step<count*4;++step) {
#pragma HLS PIPELINE II=1
        const int pixel=step/4,half=step%4;
        if(half==0)values=input.read();
        for(int j=0;j<4;++j) {
#pragma HLS UNROLL
            const int lane=half*4+j;
            const int32_t sum=int32_t(uint32_t(values.range(lane*32+31,lane*32)));
            if(last) {
                uint64_t word=(pixel%2)?words[lane]:uint64_t(0);
                word|=uint64_t(uint32_t(sum))<<((pixel%2)*32);words[lane]=word;
            } else {
                uint64_t word=(pixel%8)?words[lane]:uint64_t(0);
                word|=uint64_t(uint8_t(requant(sum,coefficients[lane],op.relu!=0)))<<((pixel%8)*8);
                words[lane]=word;
            }
        }
        if(half==3 && (last?(pixel%2==1):(pixel%8==7 || pixel==count-1))) {
            FeatureWords packed=0;
            for(int lane=0;lane<16;++lane) {
#pragma HLS UNROLL
                packed.range(lane*64+63,lane*64)=words[lane];
            }
            output.write(packed);
        }
    }
}
static void store_words(hls::stream<FeatureWords>& input,uint64_t output[16][768],uint64_t* results,
                       StreamOp op,int channels,int start,int count,bool last) {
#pragma HLS INLINE off
    const int plane=op.oh*op.ow,pitch=(plane+7)/8;
    if(last) {
        uint64_t final[16][64];
#pragma HLS ARRAY_PARTITION variable=final complete dim=1
#pragma HLS RESOURCE variable=final core=RAM_2P_BRAM
        for(int i=0;i<count/2;++i) {
#pragma HLS PIPELINE II=1
            const FeatureWords packed=input.read();
            for(int lane=0;lane<16;++lane) {
#pragma HLS UNROLL
                final[lane][i]=uint64_t(packed.range(lane*64+63,lane*64));
            }
        }
        for(int lane=0;lane<16;++lane)if(channels+lane<op.co) {
            for(int i=0;i<count/2;++i) {
#pragma HLS PIPELINE II=1
                results[((channels+lane)*plane+start)/2+i]=final[lane][i];
            }
        }
    } else {
        for(int i=0;i<(count+7)/8;++i) {
#pragma HLS PIPELINE II=1
            const FeatureWords packed=input.read();
            const int at=channels/16*pitch+start/8+i;
            for(int lane=0;lane<16;++lane) {
#pragma HLS UNROLL
                const uint64_t word=uint64_t(packed.range(lane*64+63,lane*64));
                if(channels+lane<op.co) {
                    output[lane][at]=word;
                }
            }
        }
    }
}
static void overlap_tile(const uint64_t input[16][768],uint64_t output[16][768],uint64_t* results,
                         const int8_t weights[16][320],const StreamRequant coeff[16],
                         StreamOp op,int bank,int channels,int start,int count,bool last) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    static hls::stream<Sums> sums("overlap_sums");
    static hls::stream<FeatureWords> words("overlap_words");
#pragma HLS STREAM variable=sums depth=32
#pragma HLS STREAM variable=words depth=4
// These shallow wide FIFOs otherwise fragment many BRAM primitives.
// Keep their capacities and backpressure while choosing distributed RAM.
#pragma HLS RESOURCE variable=sums core=FIFO_LUTRAM
#pragma HLS RESOURCE variable=words core=FIFO_LUTRAM
    mac_task(input,weights,op,bank,channels,start,count,sums);
    requant_words(sums,words,coeff,op,count,last);
    store_words(words,output,results,op,channels,start,count,last);
}

extern "C" int u02_stream(const uint64_t* a,const uint64_t* b,uint64_t* c,int m,int n,int k) {
#pragma HLS ALLOCATION instances=overlap_tile limit=1 function
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=weights depth=131072 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=features depth=12288 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=c offset=slave bundle=results depth=49152 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=c bundle=control
#pragma HLS INTERFACE s_axilite port=m bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=k bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if(k!=int(STREAM_PACKET_MAGIC) || m<1 || m>6 || n<64 || n>STREAM_MAX_PACKET_BYTES || n%64)return -1;
    uint32_t header[8];
    for(int i=0;i<4;++i){uint64_t v=a[i];header[i*2]=word32(v,0);header[i*2+1]=word32(v,1);}
    const uint32_t qbase=header[3],wbase=header[4];
    if(header[0]!=STREAM_PACKET_MAGIC || header[1]!=STREAM_PACKET_VERSION || header[2]!=uint32_t(m) ||
       header[5]!=uint32_t(n) || qbase<uint32_t(32+m*48) || qbase%64 || wbase<qbase || wbase%64 || wbase>uint32_t(n))return -2;
    StreamOp ops[6];
    for(int i=0;i<m;++i) {
        int values[12];for(int j=0;j<6;++j){uint64_t v=a[4+i*6+j];values[j*2]=int32_t(word32(v,0));values[j*2+1]=int32_t(word32(v,1));}
        StreamOp op={values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9],values[10],values[11]};
        if(!stream_op_valid(op) || (op.oh*op.ow)%2 || op.weight_offset%64 ||
           uint64_t(wbase)+uint32_t(op.weight_offset)+uint32_t(op.co*(op.kind?9:op.ci))>uint32_t(n) ||
           (i<m-1 && uint64_t(qbase)+(uint64_t(op.requant_offset)+uint32_t(op.co))*16>wbase) ||
           ((op.ci+15)/16)*((op.h*op.w+7)/8)>768 || ((op.co+15)/16)*((op.oh*op.ow+7)/8)>768)return -3;
        if(i && (ops[i-1].co!=op.ci || ops[i-1].oh!=op.h || ops[i-1].ow!=op.w))return -4;
        ops[i]=op;
    }
    if(header[6]!=uint32_t(ops[0].ci*ops[0].h*ops[0].w) || header[7]!=uint32_t(ops[m-1].co*ops[m-1].oh*ops[m-1].ow))return -5;
    static uint64_t data[2][16][768];
#pragma HLS ARRAY_PARTITION variable=data complete dim=1
#pragma HLS ARRAY_PARTITION variable=data complete dim=2
#pragma HLS RESOURCE variable=data core=RAM_2P_BRAM
    static uint64_t weight_words[12800];
#pragma HLS RESOURCE variable=weight_words core=RAM_2P_BRAM
    static uint64_t requant_words_cache[640];
#pragma HLS RESOURCE variable=requant_words_cache core=RAM_2P_BRAM
    int8_t weights[16][320];StreamRequant coeff[16];
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=16 dim=2
#pragma HLS RESOURCE variable=weights core=RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable=coeff complete
    const int plane=ops[0].h*ops[0].w,pitch=(plane+7)/8;
    // Read the packed tensor once as a contiguous AXI burst. Reuse the weight
    // staging RAM before any weights are loaded; channel-boundary realignment
    // below uses only local RAM, including planes that are not 8-byte aligned.
    const int input_words=(header[6]+7)/8;
    for(int i=0;i<input_words;++i) {
#pragma HLS PIPELINE II=1
        weight_words[i]=b[i];
    }
    for(int ch=0;ch<ops[0].ci;++ch)for(int j=0;j<pitch;++j) {
#pragma HLS PIPELINE II=1
        const int offset=ch*plane+j*8,shift=(offset%8)*8;
        uint64_t v=weight_words[offset/8]>>shift;
        if(shift && offset/8+1<input_words)v|=weight_words[offset/8+1]<<(64-shift);
        data[0][ch%16][ch/16*pitch+j]=v;
    }
    int bank=0;
    for(int index=0;index<m;++index) {
        const StreamOp op=ops[index];const bool last=index==m-1;
        const int kernel=op.kind?9:op.ci,weight_count=op.co*kernel;
        for(int i=0;i<(weight_count+7)/8;++i) {
#pragma HLS PIPELINE II=1
            weight_words[i]=a[(wbase+op.weight_offset)/8+i];
        }
        // The final operator returns raw INT32 and has no requant section.
        // Bounds were checked above; at most 320 channels occupy 640 words.
        // Keep validation outside this loop so HLS can infer a contiguous burst.
        if(!last) {
            const int first=qbase/8+op.requant_offset*2;
            for(int i=0;i<op.co*2;++i) {
#pragma HLS PIPELINE II=1
                requant_words_cache[i]=a[first+i];
            }
        }
        for(int channels=0;channels<op.co;channels+=16) {
            unpack_weights(weight_words,weights,channels,op.co,kernel);
            for(int lane=0;lane<16;++lane) {
                StreamRequant q={0,0,0};
                if(!last && channels+lane<op.co) {
                    const int at=(channels+lane)*2;
                    const uint64_t lo=requant_words_cache[at];q.multiplier=int32_t(lo);q.shift=int32_t(lo>>32);q.bias=int64_t(requant_words_cache[at+1]);
                    if(q.multiplier<0 || q.multiplier>=8388608 || q.shift<0 || q.shift>40 || q.bias<-(int64_t(1)<<60) || q.bias>(int64_t(1)<<60))return -6;
                }
                coeff[lane]=q;
            }
            for(int start=0;start<op.oh*op.ow;start+=128) {
                const int count=op.oh*op.ow-start<128?op.oh*op.ow-start:128;
                if(bank==0)overlap_tile(data[0],data[1],c,weights,coeff,op,0,channels,start,count,last);
                else overlap_tile(data[1],data[0],c,weights,coeff,op,1,channels,start,count,last);
            }
        }
        bank=1-bank;
    }
    return 0;
}
