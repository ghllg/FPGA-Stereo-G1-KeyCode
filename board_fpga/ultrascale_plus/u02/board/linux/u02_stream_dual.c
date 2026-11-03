// Experimental two-context transport. No new hardware or DMA allocation size.
// The shared resident implementation serializes every PL submission, retains
// the module/DMA on timeout, and refuses new RUN/mmap until safe recovery.
#include "u02_stream_dual_api.h"
#define U02_RESIDENT_DEVICE_NAME "u02_stream_dual"
#define U02_RESIDENT_ABI U02_STREAM_DUAL_ABI
#define U02_RESIDENT_SLOTS_VALID(j) u02_stream_dual_slots_valid(j)
#include "u02_stream_resident.c"
