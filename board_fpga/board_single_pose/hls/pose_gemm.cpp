#include "pose_gemm.h"

extern "C" int pose_gemm(const int8_t* a, const int8_t* b, int32_t* c,
                          int m, int n, int k) {
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=weights depth=2359296
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=features depth=117964800
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
    // Reuse a weight tile across 256 columns and load consecutive DDR rows.
    const int TM = 16, TN = 8, TK = 64, BN = 256;
    int8_t at[TM][TK], bt[TN][TK][BN/TN];
    int32_t acc[TM][TN][BN/TN];
#pragma HLS ARRAY_PARTITION variable=at complete dim=1
#pragma HLS ARRAY_PARTITION variable=bt complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS RESOURCE variable=acc core=RAM_2P_LUTRAM
    for (int row = 0; row < m; row += TM) {
        const int rows = m-row < TM ? m-row : TM;
        for (int col = 0; col < n; col += BN) {
            const int cols = n-col < BN ? n-col : BN;
            const int groups = (cols+TN-1)/TN;
            for (int block = 0; block < groups; ++block) {
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
                for (int i = 0; i < rows; ++i) {
#pragma HLS LOOP_FLATTEN off
                    for (int q = 0; q < count; ++q) {
#pragma HLS PIPELINE II=1
                        at[i][q] = a[(row+i)*k+base+q];
                    }
                }
                for (int q = 0; q < count; ++q) {
#pragma HLS LOOP_FLATTEN off
                    for (int j = 0; j < cols; ++j) {
#pragma HLS PIPELINE II=1
                        bt[j%TN][q][j/TN] = b[(base+q)*n+col+j];
                    }
                }
                for (int q = 0; q < count; ++q) {
#pragma HLS LOOP_FLATTEN off
                    for (int block = 0; block < groups; ++block) {
#pragma HLS PIPELINE II=1
                        int8_t av[TM], bv[TN];
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
#pragma HLS RESOURCE variable=product core=Mul
                                    acc[i][j][block] += product;
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < rows; ++i) {
#pragma HLS LOOP_FLATTEN off
                for (int j = 0; j < cols; ++j) {
#pragma HLS PIPELINE II=1
                    c[(row+i)*n+col+j] = acc[i][j%TN][j/TN];
                }
            }
        }
    }
    return 0;
}
