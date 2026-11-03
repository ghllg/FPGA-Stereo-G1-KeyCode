#ifndef U02_REGION_SESSION_H
#define U02_REGION_SESSION_H
#include "u02_region_api.h"
#include "u02_region_binding_validation.h"
#ifdef __KERNEL__
#include <linux/errno.h>
#else
#include <errno.h>
#endif

/* Serialized by the real driver's mutex. Callbacks allocate private owned
 * DMA, copy_from_user and free it. Neither config buffer may be mmap'ed.
 * This core does not access registers or establish that hardware is idle. */
struct u02_region_buffer {unsigned char *cpu;u02r_u64 dma;size_t capacity;};
struct u02_region_config {
    struct u02_region_buffer metadata,model;
    u02r_u32 metadata_bytes,model_bytes,arena_bytes;
    u02r_u64 token;
};
struct u02_region_session {struct u02_region_config active;u02r_u64 generation;unsigned inflight;};
struct u02_region_memory_ops {
    int (*allocate)(void *context,size_t bytes,struct u02_region_buffer *buffer);
    void (*release)(void *context,struct u02_region_buffer *buffer);
    int (*copy_user)(void *context,void *destination,u02r_u64 source,size_t bytes);
};
static inline void u02_region_free_config(struct u02_region_config *config,
        const struct u02_region_memory_ops *ops,void *context)
{
    if (config->metadata.cpu) ops->release(context,&config->metadata);
    if (config->model.cpu) ops->release(context,&config->model);
    memset(config,0,sizeof(*config));
}
static inline int u02_region_install_config(struct u02_region_session *session,
        struct u02_region_install *request,const struct u02_region_memory_ops *ops,
        void *context,struct u02_region_validation_workspace *workspace)
{
    struct u02_region_config replacement={0},previous;
    int status;
    if (!session || !request || !ops || !ops->allocate || !ops->release || !ops->copy_user || !workspace) return -EINVAL;
    if (session->inflight) return -EBUSY;
    if (!u02_region_install_valid(request)) return -EINVAL;
    if (session->generation==~(u02r_u64)0) return -EOVERFLOW;
    replacement.metadata_bytes=request->metadata_bytes;replacement.model_bytes=request->model_bytes;
    status=ops->allocate(context,request->metadata_bytes,&replacement.metadata);
    if (status) goto fail;
    if (!replacement.metadata.cpu || replacement.metadata.capacity<request->metadata_bytes) {status=-ENOMEM;goto fail;}
    status=ops->allocate(context,request->model_bytes,&replacement.model);
    if (status) goto fail;
    if (!replacement.model.cpu || replacement.model.capacity<request->model_bytes) {status=-ENOMEM;goto fail;}
    status=ops->copy_user(context,replacement.metadata.cpu,request->metadata_ptr,request->metadata_bytes);
    if (status) goto fail;
    status=ops->copy_user(context,replacement.model.cpu,request->model_ptr,request->model_bytes);
    if (status) goto fail;
    status=u02_region_binding_validate(replacement.metadata.cpu,request->metadata_bytes,
        replacement.model.cpu,request->model_bytes,workspace);
    if (status) {status=-EINVAL;goto fail;}
    replacement.arena_bytes=u02r_word(replacement.metadata.cpu,20);
    replacement.token=session->generation+1;
    previous=session->active;session->active=replacement;session->generation=replacement.token;
    request->token=replacement.token;
    u02_region_free_config(&previous,ops,context);
    return 0;
fail:
    u02_region_free_config(&replacement,ops,context);
    return status;
}
static inline int u02_region_begin(struct u02_region_session *session,const struct u02_region_job *job)
{
    if (!session || !job) return -EINVAL;
    if (session->inflight) return -EBUSY;
    if (!session->active.token || !u02_region_job_valid(job,session->active.token)) return -EINVAL;
    session->inflight=1;
    return 0;
}
static inline int u02_region_observed_idle(struct u02_region_session *session,int idle)
{
    if (!session) return -EINVAL;
    /* idle must come from real, independently verified IP status, not a user
     * argument or a timeout. The driver syncs scratch to CPU before this call. */
    if (!idle) return -EBUSY;
    session->inflight=0;
    return 0;
}
static inline int u02_region_close_config(struct u02_region_session *session,
        const struct u02_region_memory_ops *ops,void *context)
{
    if (!session || !ops || !ops->release) return -EINVAL;
    if (session->inflight) return -EBUSY;
    u02_region_free_config(&session->active,ops,context);
    return 0;
}
#endif
