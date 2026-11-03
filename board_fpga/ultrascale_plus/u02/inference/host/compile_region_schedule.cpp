#define U02_STREAM_REGION_HOOK 1
#include "stream_schedule.h"
// Compile-only against the actual NCNN headers and generated complete graph.
void compile_region_schedule(StreamSchedule& schedule,RegionScheduleExtension& region,
        const ncnn::Mat& input,ncnn::Mat& output) {
    schedule.attach_region(&region);schedule.forward(input,output);
}
