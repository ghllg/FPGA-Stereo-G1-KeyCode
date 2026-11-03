"""Create a separate ABI4 driver; never load it or mutate accepted ABI3."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)

validation = (root / 'u02_stream_resident_validation.h').read_text()
validation = replace_once(validation,
    'j->b_offset != U02_STREAM_RESIDENT_B_OFFSET || j->c_offset != U02_STREAM_RESIDENT_C_OFFSET ||',
    'j->b_offset != U02_STREAM_RESIDENT_B_OFFSET ||\n'
    '        (j->c_offset & (U02_STREAM_FP32_C_ALIGNMENT - 1)) ||\n'
    '        j->c_offset < U02_STREAM_FP32_C_BASE ||\n'
    '        (u64)j->c_offset >= (u64)U02_STREAM_FP32_C_BASE + U02_STREAM_FP32_C_POOL_BYTES ||\n'
    '        (u64)j->c_offset + U02_STREAM_OUTPUT_BYTES + 64 > U02_POSE_ARENA_BYTES ||')
validation = replace_once(validation,
    '!j->c_bytes || j->c_bytes > U02_STREAM_OUTPUT_BYTES ||',
    '!j->c_bytes || j->c_bytes > U02_STREAM_OUTPUT_BYTES ||\n'
    '        (u64)j->c_offset + j->c_bytes > (u64)U02_STREAM_FP32_C_BASE + U02_STREAM_FP32_C_POOL_BYTES ||')
validation = replace_once(validation, 'h[1] != 1', 'h[1] != U02_STREAM_FP32_VERSION')
validation = replace_once(validation,
    '        if (i + 1 < j->m) {\n'
    '            if ((u64)h[3] + ((u64)op->requant_offset + op->co) * 16 > h[4]) return -EINVAL;',
    '        if ((u64)h[3] + ((u64)op->requant_offset + op->co) * 16 > h[4]) return -EINVAL;\n'
    '        if (i + 1 < j->m) {')
validation = replace_once(validation,
    '            }\n        }\n    }\n    if (h[6]',
    '            }\n        } else {\n'
    '            for (ch = 0; ch < op->co; ++ch) {\n'
    '                struct u02_stream_fp32_affine q;\n'
    '                memcpy(&q, (u8 *)arena + j->a_offset + h[3] + ((u64)op->requant_offset + ch) * 16, sizeof(q));\n'
    '                if (!u02_stream_fp32_affine_valid(&q)) return -EINVAL;\n'
    '            }\n        }\n    }\n    if (h[6]')
driver = (root / 'u02_stream_resident.c').read_text()
driver = driver.replace('u02_stream_resident', 'u02_stream_fp32')
driver = replace_once(driver, 'U02_STREAM_RESIDENT_ABI', 'U02_STREAM_FP32_ABI')
driver = replace_once(driver,
    'All fixed slot bases and capacities are page aligned.',
    'Packet/input bases are page aligned; output starts are cache-line aligned.')
driver = replace_once(driver,
    'U02 fused TinyPose streaming transport using guarded resident packet starts in kernel-owned DMA',
    'U02 packet2 FP32-output TinyPose transport with guarded resident packets and output starts in kernel-owned DMA')
for name, text in [('u02_stream_fp32_validation.h', validation), ('u02_stream_fp32.c', driver)]:
    destination = root / name
    assert not destination.exists(), 'Preserve previous preparation'
    destination.write_text(text, newline='\n')
    print(name, hashlib.sha256(destination.read_bytes()).hexdigest())
