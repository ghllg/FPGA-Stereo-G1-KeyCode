#ifndef U02_REGION_DUAL_PACKET_VALIDATION_H
#define U02_REGION_DUAL_PACKET_VALIDATION_H
/* Generated from formal ABI4 validation. Only private stable bytes may enter.
 * Caller must copy exactly packet_bytes bytes before invoking this function. */
#include "u02_region_model_validation.h"
#ifdef __KERNEL__
#include <linux/errno.h>
#include <linux/string.h>
#else
#include <errno.h>
#include <string.h>
#endif
struct u02_dual_op {u02r_s32 kind,ci,co,h,w,oh,ow,stride,pad,relu,weight_offset,requant_offset;};
struct u02_dual_requant {u02r_s32 multiplier,shift;u02r_s64 bias;};
struct u02_dual_affine {u02r_u32 incoming,weight,bias,reserved;};
static inline int u02_dual_packet_validate(const unsigned char *packet,
        u02r_u32 packet_bytes,u02r_u32 input_bytes,u02r_u32 output_bytes,u02r_u32 operators)
{
    u02r_u32 h[8];struct u02_dual_op ops[6];unsigned i,ch;
    if (!packet || operators<1 || operators>6 || packet_bytes<64 ||
        packet_bytes>1048576 || packet_bytes%64 || !input_bytes || input_bytes>98304 ||
        !output_bytes || output_bytes>393216) return -EINVAL;
    for (i=0;i<8;++i) h[i]=u02r_word(packet,i*4);
    if (h[0] != 0x55303253u || h[1] != 2u || h[2] != operators || h[5] != packet_bytes ||
        h[3] < 32 + operators * sizeof(ops[0]) || (h[3] & 63) || h[4] < h[3] ||
        (h[4] & 63) || h[4] > packet_bytes || h[6] != input_bytes || (u02r_u64)h[7] * 4 != output_bytes)
        return -EINVAL;
    memcpy(ops,packet+32,operators*sizeof(ops[0]));
    for (i = 0; i < operators; ++i) {
        const struct u02_dual_op *op = &ops[i];
        u02r_u32 weights;
        if ((op->kind != 0 && op->kind != 1) || op->ci < 1 || op->ci > 320 ||
            op->co < 1 || op->co > 320 || op->h < 1 || op->h > 64 || op->w < 1 || op->w > 48 ||
            op->oh < 1 || op->oh > 64 || op->ow < 1 || op->ow > 48 ||
            (op->stride != 1 && op->stride != 2) || op->pad < 0 || op->pad > 1 ||
            (op->relu != 0 && op->relu != 1) || op->weight_offset < 0 ||
            (op->weight_offset & 63) || op->requant_offset < 0)
            return -EINVAL;
        if ((!op->kind && (op->stride != 1 || op->pad)) || (op->kind && op->ci != op->co) ||
            op->oh != (op->h + 2 * op->pad - (op->kind ? 3 : 1)) / op->stride + 1 ||
            op->ow != (op->w + 2 * op->pad - (op->kind ? 3 : 1)) / op->stride + 1 ||
            (op->oh * op->ow & 1) || op->ci * op->h * op->w > 98304u ||
            op->co * op->oh * op->ow > 393216u / 4 ||
            ((op->ci + 15) / 16) * ((op->h * op->w + 7) / 8) > 768 ||
            ((op->co + 15) / 16) * ((op->oh * op->ow + 7) / 8) > 768)
            return -EINVAL;
        weights = op->co * (op->kind ? 9 : op->ci);
        if ((u02r_u64)h[4] + op->weight_offset + weights > packet_bytes) return -EINVAL;
        if (i && (ops[i-1].co != op->ci || ops[i-1].oh != op->h || ops[i-1].ow != op->w)) return -EINVAL;
        if ((u02r_u64)h[3] + ((u02r_u64)op->requant_offset + op->co) * 16 > h[4]) return -EINVAL;
        if (i + 1 < operators) {
            for (ch = 0; ch < op->co; ++ch) {
                struct u02_dual_requant q;
                memcpy(&q, packet + h[3] + ((u02r_u64)op->requant_offset + ch) * 16, sizeof(q));
                if (q.multiplier < 0 || q.multiplier >= 8388608 || q.shift < 0 || q.shift > 40 ||
                    q.bias < -(1LL << 60) || q.bias > (1LL << 60)) return -EINVAL;
            }
        } else {
            for (ch = 0; ch < op->co; ++ch) {
                struct u02_dual_affine q;
                memcpy(&q, packet + h[3] + ((u02r_u64)op->requant_offset + ch) * 16, sizeof(q));
                if ((!u02r_scale_valid(q.incoming) || !u02r_scale_valid(q.weight) || !u02r_bias_valid(q.bias) || q.reserved)) return -EINVAL;
            }
        }
    }
    if (h[6] != ops[0].ci * ops[0].h * ops[0].w ||
        h[7] != ops[operators-1].co * ops[operators-1].oh * ops[operators-1].ow) return -EINVAL;
    return 0;
}

#endif
