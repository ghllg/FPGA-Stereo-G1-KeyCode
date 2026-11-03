/* Packet validation plus guarded resident packet starts keep all hardware accesses
 * inside kernel-owned memory, even if writable userspace metadata changes. */
static int validate_job(const struct u02_pose_job *j)
{
    u32 h[8];
    struct u02_stream_op ops[6];
    unsigned int i, ch;
#ifdef U02_FP32_SLOTS_VALID
    if (!U02_FP32_SLOTS_VALID(j) ||
#else
    if ((j->a_offset & (U02_STREAM_RESIDENT_ALIGNMENT - 1)) ||
        j->a_offset >= U02_STREAM_RESIDENT_POOL_BYTES ||
        (u64)j->a_offset + U02_STREAM_PACKET_BYTES > U02_STREAM_RESIDENT_A_REGION_BYTES ||
        j->b_offset != U02_STREAM_RESIDENT_B_OFFSET ||
        (j->c_offset & (U02_STREAM_FP32_C_ALIGNMENT - 1)) ||
        j->c_offset < U02_STREAM_FP32_C_BASE ||
        (u64)j->c_offset >= (u64)U02_STREAM_FP32_C_BASE + U02_STREAM_FP32_C_POOL_BYTES ||
        (u64)j->c_offset + U02_STREAM_OUTPUT_BYTES + 64 > U02_POSE_ARENA_BYTES ||
#endif
        j->m < 1 || j->m > 6 || j->k != U02_STREAM_MAGIC || j->n < 64 ||
        j->n > U02_STREAM_PACKET_BYTES || (j->n & 63) || j->a_bytes != j->n ||
        !j->b_bytes || j->b_bytes > U02_STREAM_INPUT_BYTES ||
        !j->c_bytes || j->c_bytes > U02_STREAM_OUTPUT_BYTES ||
#ifndef U02_FP32_SLOTS_VALID
        (u64)j->c_offset + j->c_bytes > (u64)U02_STREAM_FP32_C_BASE + U02_STREAM_FP32_C_POOL_BYTES ||
#endif
        !j->timeout_us || j->timeout_us > 30000000)
        return -EINVAL;
    memcpy(h, (u8 *)arena + j->a_offset, sizeof(h));
    if (h[0] != U02_STREAM_MAGIC || h[1] != U02_STREAM_FP32_VERSION || h[2] != j->m || h[5] != j->n ||
        h[3] < 32 + j->m * sizeof(ops[0]) || (h[3] & 63) || h[4] < h[3] ||
        (h[4] & 63) || h[4] > j->n || h[6] != j->b_bytes || (u64)h[7] * 4 != j->c_bytes)
        return -EINVAL;
    memcpy(ops, (u8 *)arena + j->a_offset + 32, j->m * sizeof(ops[0]));
    for (i = 0; i < j->m; ++i) {
        const struct u02_stream_op *op = &ops[i];
        u32 weights;
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
            (op->oh * op->ow & 1) || op->ci * op->h * op->w > U02_STREAM_INPUT_BYTES ||
            op->co * op->oh * op->ow > U02_STREAM_OUTPUT_BYTES / 4 ||
            ((op->ci + 15) / 16) * ((op->h * op->w + 7) / 8) > 768 ||
            ((op->co + 15) / 16) * ((op->oh * op->ow + 7) / 8) > 768)
            return -EINVAL;
        weights = op->co * (op->kind ? 9 : op->ci);
        if ((u64)h[4] + op->weight_offset + weights > j->n) return -EINVAL;
        if (i && (ops[i-1].co != op->ci || ops[i-1].oh != op->h || ops[i-1].ow != op->w)) return -EINVAL;
        if ((u64)h[3] + ((u64)op->requant_offset + op->co) * 16 > h[4]) return -EINVAL;
        if (i + 1 < j->m) {
            for (ch = 0; ch < op->co; ++ch) {
                struct u02_stream_requant q;
                memcpy(&q, (u8 *)arena + j->a_offset + h[3] + ((u64)op->requant_offset + ch) * 16, sizeof(q));
                if (q.multiplier < 0 || q.multiplier >= 8388608 || q.shift < 0 || q.shift > 40 ||
                    q.bias < -(1LL << 60) || q.bias > (1LL << 60)) return -EINVAL;
            }
        } else {
            for (ch = 0; ch < op->co; ++ch) {
                struct u02_stream_fp32_affine q;
                memcpy(&q, (u8 *)arena + j->a_offset + h[3] + ((u64)op->requant_offset + ch) * 16, sizeof(q));
                if (!u02_stream_fp32_affine_valid(&q)) return -EINVAL;
            }
        }
    }
    if (h[6] != ops[0].ci * ops[0].h * ops[0].w ||
        h[7] != ops[j->m-1].co * ops[j->m-1].oh * ops[j->m-1].ow) return -EINVAL;
    return 0;
}
