#ifndef U02_REGION_BINDING_VALIDATION_H
#define U02_REGION_BINDING_VALIDATION_H
#include "u02_region_program_validation.h"

/* Both buffers must be private snapshots throughout validation and execution.
 * No kernel FP, DMA, MMIO or user pointer dereference occurs here. */
static inline int u02_region_binding_validate(const void *metadata,size_t metadata_bytes,
        const void *model,size_t model_bytes,struct u02_region_validation_workspace *workspace)
{
    const unsigned char *program=(const unsigned char *)metadata;
    const unsigned char *weights=(const unsigned char *)model;
    U02rRegionHeader h;
    unsigned i,j,count;
    int status=u02_region_program_validate(metadata,metadata_bytes,workspace);
    if (status) return status;
    status=u02_region_model_validate(model,model_bytes);
    if (status) return status;
    memcpy(&h,program,sizeof(h));count=u02r_word(weights,12);
    if (count!=h.stream_commands) return -16;
    for (i=0;i<h.commands;++i) {
        U02rRegionCommand q;int found=0;
        memcpy(&q,program+h.command_offset+i*32,32);
        if (q.opcode!=U02R_REGION_STREAM) continue;
        for (j=0;j<count;++j) if (u02r_word(weights,64+j*64)==q.group) {
            size_t packet=u02r_word(weights,64+j*64+4);
            unsigned operations=u02r_word(weights,packet+8);
            size_t first=packet+32,last=packet+32+(operations-1)*48;
            const U02rRegionTensor *input=&workspace->tensors[q.input0],*output=&workspace->tensors[q.output0];
            if (q.param!=u02r_word(weights,first) || input->c!=u02r_word(weights,first+4) ||
                input->h!=u02r_word(weights,first+12) || input->w!=u02r_word(weights,first+16) ||
                output->c!=u02r_word(weights,last+8) || output->h!=u02r_word(weights,last+20) ||
                output->w!=u02r_word(weights,last+24)) return -17;
            found=1;break;
        }
        if (!found) return -16;
    }
    return 0;
}
#endif
