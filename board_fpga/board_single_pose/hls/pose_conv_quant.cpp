#include "conv_contract.h"
#include "quantize_stream.h"

extern "C" int pose_conv_quant(const uint64_t* a, uint64_t* b, int32_t* c,
int m, int n, int k) {
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=weights depth=294912 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=features depth=14745600 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=c offset=slave bundle=results depth=13107200
#pragma HLS INTERFACE s_axilite port=a bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=b bundle=control offset=0x1c
#pragma HLS INTERFACE s_axilite port=c bundle=control offset=0x28
#pragma HLS INTERFACE s_axilite port=m bundle=control offset=0x34
#pragma HLS INTERFACE s_axilite port=n bundle=control offset=0x3c
#pragma HLS INTERFACE s_axilite port=k bundle=control offset=0x44
#pragma HLS INTERFACE s_axilite port=return bundle=control
    const bool convolution=(k & POSE_CONV_FLAG)!=0;
    if (convolution) k-=POSE_CONV_FLAG;
    if (m < 1 || m > POSE_MAX_M || n < 1 || n > POSE_MAX_N ||
    k < 1 || k > POSE_MAX_K) return -1;
    PoseConv shape={};
    int image_word=8;
    if (convolution) {
        uint32_t d[16];
        for (int i=0;i<8;++i) {
#pragma HLS PIPELINE II=1
            const uint64_t word=b[i];
            d[i*2]=uint32_t(word);d[i*2+1]=uint32_t(word>>32);
        }
        shape.w=d[1];shape.h=d[2];shape.channels=d[3];shape.kw=d[4];
        shape.kh=d[5];shape.sw=d[6];shape.sh=d[7];shape.left=d[8];
        shape.top=d[9];shape.ow=d[10];shape.oh=d[11];shape.start=d[12];
        if (d[0]!=POSE_CONV_MAGIC || d[13]>2 || d[14] || d[15] ||
            !pose_conv_valid(shape,n,k)) return -2;
        if (d[13]) {
            const int count=shape.w*shape.h*shape.channels;
            if (count>1048576) return -3;
            // Relative to b+8: thresholds occupy 64 words; FP32 input and
            // INT8 workspace each start on a 64-byte cache-line boundary.
            const int output_word=64+((count*4+63)/64)*8;
            if (d[13]==1 && pose_quantize_stream(b+8,count,64,output_word)) return -4;
            image_word=8+output_word;
        }
    }
    // Fixed lane banks avoid dynamic partition selection during the MAC loop.
    // Cache all partial outputs for 256 rows x 128 columns in BRAM. Load each
    // feature tile once for all 16 output-row tiles, retaining the 128 MAC lanes.
    const int TM = 16, TN = 8, TK = 64, BN = 128, BM = 256;
    uint64_t at[TM][TK/8];
    int8_t bt[TN][TK][BN/TN];
    int32_t acc[TM][TN][(BM/TM)*(BN/TN)];
    // Four replicas provide eight read ports for eight output lanes.
    // Each cache covers at most 2048 consecutive source bytes.
    uint64_t image_cache[4][256];
#pragma HLS ARRAY_PARTITION variable=image_cache complete dim=1
#pragma HLS RESOURCE variable=image_cache core=RAM_2P_BRAM
    int16_t source_x[TN][BN/TN],source_y[TN][BN/TN];
    int32_t source_row[TN][BN/TN];
#pragma HLS ARRAY_PARTITION variable=source_x complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_y complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_row complete dim=1
    uint64_t packed_a[TK/8+1], packed_b[BN/8+1];
#pragma HLS RESOURCE variable=packed_b core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=at complete dim=1
#pragma HLS ARRAY_PARTITION variable=bt complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS RESOURCE variable=acc core=RAM_2P_BRAM
    for (int row_base = 0; row_base < m; row_base += BM) {
        const int all_rows = m-row_base < BM ? m-row_base : BM;
        const int tiles=(all_rows+TM-1)/TM;
        for (int col = 0; col < n; col += BN) {
            const int cols = n-col < BN ? n-col : BN;
            const int groups = (cols+TN-1)/TN;
            for (int block = 0; block < tiles*(BN/TN); ++block) {
#pragma HLS PIPELINE II=1
                for (int i = 0; i < TM; ++i) {
#pragma HLS UNROLL
                    for (int j = 0; j < TN; ++j) {
#pragma HLS UNROLL
                        acc[i][j][block] = 0;
                    }
                }
            }
            if (convolution) {
                int y=(shape.start+col)/shape.ow;
                int x=shape.start+col-y*shape.ow;
                // Initialize inactive tail lanes too; gather masks them out.
                for (int j=0;j<groups*TN;++j) {
#pragma HLS PIPELINE II=1
                    const int16_t sx=(shape.sw==2 ? x*2 : x)-shape.left;
                    const int16_t sy=(shape.sh==2 ? y*2 : y)-shape.top;
                    source_x[j%TN][j/TN]=sx;
                    source_y[j%TN][j/TN]=sy;
                    source_row[j%TN][j/TN]=sy*int16_t(shape.w);
                    ++x;
                    if (x==shape.ow) {x=0;++y;}
                }
            }
            for (int base = 0; base < k; base += TK) {
                const int count = k-base < TK ? k-base : TK;
                if (!convolution) {
                for (int q = 0; q < count; ++q) {
#pragma HLS LOOP_FLATTEN off
                    const int first=(base+q)*n+col;
                    const int offset=first&7, words=(offset+cols+7)/8;
                    for (int w=0;w<words;++w) {
#pragma HLS PIPELINE II=1
                        packed_b[w]=b[first/8+w];
                    }
                    for (int block = 0; block < groups; ++block) {
#pragma HLS PIPELINE II=1
                        uint64_t word=packed_b[block];
                        if (offset) {
                            uint64_t next=block+1<words ? packed_b[block+1] : 0;
                            word=(word>>(offset*8)) | (next<<((8-offset)*8));
                        }
                        for (int lane=0;lane<TN;++lane) {
#pragma HLS UNROLL
                            bt[lane][q][block]=int8_t(word>>(lane*8));
                        }
                    }
                }
                } else {
                    const int first_pixel=shape.start+col;
                    const int last_pixel=first_pixel+cols-1;
                    const int first_y=first_pixel/shape.ow;
                    const int first_x=first_pixel-first_y*shape.ow;
                    const int last_y=last_pixel/shape.ow;
                    const int last_x=last_pixel-last_y*shape.ow;
                    int cached_channel=-1,cached_first=-1,cached_last=-1;
                    for (int q=0;q<count;++q) {
#pragma HLS LOOP_FLATTEN off
                        const int reduction=base+q;
                        const int plane=shape.kw*shape.kh;
                        const int channel=plane==9 ? reduction/9 : (plane==3 ? reduction/3 : reduction);
                        const int kernel=reduction-channel*plane;
                        const int ky=shape.kw==3 ? kernel/3 : kernel;
                        const int kx=shape.kw==3 ? kernel-ky*3 : 0;
                        const int y0=first_y*shape.sh-shape.top+ky;
                        const int y1=last_y*shape.sh-shape.top+ky;
                        const int x0=first_x*shape.sw-shape.left+kx;
                        const int x1=last_x*shape.sw-shape.left+kx;
                        const int total=shape.w*shape.h;
                        int low=y0*shape.w+(x0<0 ? 0 : (x0>shape.w ? shape.w : x0));
                        int high=y1*shape.w+(x1<0 ? -1 : (x1>=shape.w ? shape.w-1 : x1));
                        if (y0<0) low=0;
                        if (y1>=shape.h) high=total-1;
                        const bool visible=low<=high && low<total && high>=0;
                        const int channel_base=channel*total;
                        const int word0=visible ? (channel_base+low)/8 : 0;
                        const int word1=visible ? (channel_base+high)/8 : -1;
                        const int words=word1-word0+1;
                        if (words>256) return -3;
                        if (channel!=cached_channel || word0!=cached_first || word1!=cached_last) {
                            for (int w=0;w<words;++w) {
#pragma HLS PIPELINE II=1
                                const uint64_t word=b[image_word+word0+w];
                                for (int copy=0;copy<4;++copy) {
#pragma HLS UNROLL
                                    image_cache[copy][w]=word;
                                }
                            }
                            cached_channel=channel;cached_first=word0;cached_last=word1;
                        }
                        const int channel_kernel_row=channel_base+ky*shape.w;
                        for (int block=0;block<groups;++block) {
#pragma HLS PIPELINE II=1
                            for (int lane=0;lane<TN;++lane) {
#pragma HLS UNROLL
                                const int sx=source_x[lane][block]+kx;
                                const int sy=source_y[lane][block]+ky;
                                int8_t value=0;
                                if (block*TN+lane<cols && sx>=0 && sx<shape.w && sy>=0 && sy<shape.h) {
                                    const int address=channel_kernel_row+source_row[lane][block]+sx;
                                    const uint64_t word=image_cache[lane/2][address/8-word0];
                                    value=int8_t(word>>((address&7)*8));
                                }
                                bt[lane][q][block]=value;
                            }
                        }
                    }
                }
                for(int tile=0;tile<tiles;++tile) {
                    const int row=row_base+tile*TM;
                    const int rows=all_rows-tile*TM < TM ? all_rows-tile*TM : TM;
                    for (int i = 0; i < rows; ++i) {
#pragma HLS LOOP_FLATTEN off
                        const int first=(row+i)*k+base;
                        const int offset=first&7, words=(offset+count+7)/8;
                        for (int w=0;w<words;++w) {
#pragma HLS PIPELINE II=1
                            packed_a[w]=a[first/8+w];
                        }
                        for (int block=0;block<(count+7)/8;++block) {
#pragma HLS PIPELINE II=1
                            uint64_t word=packed_a[block];
                            if (offset) {
                                const uint64_t next=block+1<words ? packed_a[block+1] : 0;
                                word=(word>>(offset*8)) | (next<<((8-offset)*8));
                            }
                            at[i][block]=word;
                        }
                    }
                    for (int q = 0; q < count; ++q) {
#pragma HLS LOOP_FLATTEN off
                        // Extract the 16 weight bytes once per reduction step,
                        // outside the accumulation loop's timing path.
                        int8_t av[TM];
#pragma HLS ARRAY_PARTITION variable=av complete
                        for (int i=0;i<TM;++i) {
#pragma HLS UNROLL
                            av[i]=i<rows ? int8_t(at[i][q/8] >> ((q&7)*8)) : 0;
                        }
                        for (int block = 0; block < groups; ++block) {
#pragma HLS PIPELINE II=1
                            int8_t bv[TN];
#pragma HLS DEPENDENCE variable=acc inter false
                            // Inner iterations address distinct column groups.
                            // The outer q loop is not pipelined and must stay so:
                            // its pipeline drains before revisiting these addresses.
#pragma HLS ARRAY_PARTITION variable=bv complete
                            for (int j = 0; j < TN; ++j) {
#pragma HLS UNROLL
                                bv[j] = block*TN+j < cols ? bt[j][q][block] : 0;
                            }
                            for (int i = 0; i < TM; ++i) {
#pragma HLS UNROLL
                                for (int j = 0; j < TN; ++j) {
#pragma HLS UNROLL
                                    int product = av[i] * bv[j];
#pragma HLS RESOURCE variable=product core=Mul
                                    acc[i][j][tile*(BN/TN)+block] += product;
                                }
                            }
                        }
                    }
                } // row tile
            } // K tile
            for (int i = 0; i < all_rows; ++i) {
#pragma HLS LOOP_FLATTEN off
                for (int j = 0; j < cols; ++j) {
#pragma HLS PIPELINE II=1
                    c[(row_base+i)*n+col+j] = acc[i%TM][j%TN][(i/TM)*(BN/TN)+j/TN];
                }
            }
        }
    }
    return 0;
}
