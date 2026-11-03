#include "pose_gemm_reuse.h"

extern "C" int pose_gemm_reuse_pipe(const uint64_t* a, const uint64_t* b, int32_t* c,
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
    if (m < 1 || m > POSE_MAX_M || n < 1 || n > POSE_MAX_N ||
    k < 1 || k > POSE_MAX_K) return -1;
    // Fixed lane banks avoid dynamic partition selection during the MAC loop.
    // Cache all partial outputs for 256 rows x 128 columns in BRAM. Load each
    // feature tile once for all 16 output-row tiles, retaining the 128 MAC lanes.
    const int TM = 16, TN = 8, TK = 64, BN = 128, BM = 256;
    int8_t at[TM][TK], bt[TN][TK][BN/TN];
    int32_t acc[TM][TN][(BM/TM)*(BN/TN)];
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
            for (int base = 0; base < k; base += TK) {
                const int count = k-base < TK ? k-base : TK;
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
                        for (int q=0;q<count;++q) {
#pragma HLS PIPELINE II=1
                            at[i][q]=int8_t(packed_a[(offset+q)/8] >> (((offset+q)&7)*8));
                        }
                    }
                    for (int q = 0; q < count; ++q) {
#pragma HLS LOOP_FLATTEN off
                        for (int block = 0; block < groups; ++block) {
#pragma HLS PIPELINE II=1
                            int8_t av[TM], bv[TN];
#pragma HLS DEPENDENCE variable=acc inter false
                            // Inner iterations address distinct column groups.
                            // The outer q loop is not pipelined and must stay so:
                            // its pipeline drains before revisiting these addresses.
#pragma HLS ARRAY_PARTITION variable=av complete
#pragma HLS ARRAY_PARTITION variable=bv complete
                            for (int i = 0; i < TM; ++i) {
#pragma HLS UNROLL
                                av[i] = i < rows ? at[i][q] : 0;
                            }
                            for (int j = 0; j < TN; ++j) {
#pragma HLS UNROLL
                                bv[j] = block*TN+j < cols ? bt[j][q][block] : 0;
                            }
                            for (int i = 0; i < TM; ++i) {
#pragma HLS UNROLL
                                for (int j = 0; j < TN; ++j) {
#pragma HLS UNROLL
                                    int product = av[i] * bv[j];
#pragma HLS RESOURCE variable=product core=Mul latency=2
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
