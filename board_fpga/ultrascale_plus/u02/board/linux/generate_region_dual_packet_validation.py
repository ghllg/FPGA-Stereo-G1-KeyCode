"""Derive candidate private-packet checks from the accepted ABI4 validator."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    source = root/'u02_stream_fp32_validation.h'
    original = source.read_text()
    body = original[original.index('    memcpy(h,'):]
    body = body.replace('memcpy(h, (u8 *)arena + j->a_offset, sizeof(h));',
                        'for (i=0;i<8;++i) h[i]=u02r_word(packet,i*4);')
    body = body.replace('memcpy(ops, (u8 *)arena + j->a_offset + 32, j->m * sizeof(ops[0]));',
                        'memcpy(ops,packet+32,operators*sizeof(ops[0]));')
    body = body.replace('(u8 *)arena + j->a_offset', 'packet')
    for name, value in {'m':'operators', 'n':'packet_bytes', 'b_bytes':'input_bytes', 'c_bytes':'output_bytes'}.items():
        body = body.replace('j->'+name, value)
    for old, new in {'u32':'u02r_u32', 'u64':'u02r_u64',
                     'struct u02_stream_op':'struct u02_dual_op',
                     'struct u02_stream_requant':'struct u02_dual_requant',
                     'struct u02_stream_fp32_affine':'struct u02_dual_affine',
                     'U02_STREAM_MAGIC':'0x55303253u', 'U02_STREAM_FP32_VERSION':'2u',
                     'U02_STREAM_INPUT_BYTES':'98304u', 'U02_STREAM_OUTPUT_BYTES':'393216u'}.items():
        import re
        body = re.sub(r'\b'+re.escape(old)+r'\b', new, body)
    body = body.replace('!u02_stream_fp32_affine_valid(&q)',
                        '(!u02r_scale_valid(q.incoming) || !u02r_scale_valid(q.weight) || !u02r_bias_valid(q.bias) || q.reserved)')
    if 'j->' in body or 'arena' in body or 'U02_STREAM_' in body:
        raise ValueError('Incomplete private packet conversion')
    header = '''#ifndef U02_REGION_DUAL_PACKET_VALIDATION_H
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
'''
    output = root/'u02_region_dual_packet_validation.h'
    output.write_text(header+body+'\n#endif\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'u02_region_dual_packet_validation.manifest.json').write_text(json.dumps({
        'source_sha256':{p.name:digest(p) for p in (source, Path(__file__), root/'u02_region_model_validation.h')},
        'generated_sha256':digest(output), 'board_accessed':False,
        'scope':'Private packet validation only; no driver, MMIO or DMA integration.'},indent=2)+'\n')
    print('U02_REGION_DUAL_PACKET_VALIDATOR_GENERATED')


if __name__=='__main__':main()
