/* Isolated ABI9/device. Reuse the accepted ownership, cache, timeout and
 * recovery implementation without changing its allocation size or polling. */
#include "u02_stream_pipeline_api.h"
#define U02_RESIDENT_DEVICE_NAME "u02_stream_pipeline"
#define U02_RESIDENT_ABI U02_STREAM_PIPELINE_ABI
#define U02_RESIDENT_SLOTS_VALID(j) u02_stream_pipeline_slots_valid(j)
#include "u02_stream_resident.c"
