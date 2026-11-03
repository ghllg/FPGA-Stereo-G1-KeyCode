#pragma once
#include <vector>
#include <cstring>
#include <cmath>
#include <cfenv>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include "stream_range_hints_generated.h"

inline uint64_t stream_range_fpcr() {
#ifdef __aarch64__
    uint64_t value;asm volatile("mrs %0, fpcr":"=r"(value));return value;
#else
    return uint64_t(std::fegetround());
#endif
}

// Private to one Backend/model. Entries contain scalar metadata, never tensor
// pointers. Per-channel producer writes join before dependent layer reads.
class StreamRangeCache {
    struct Entry {float maximum=0;uint32_t epoch=0;uint64_t fpcr=0;};
    std::vector<Entry> entries;
    bool registered[110]={};
    bool enabled=false;
    uint32_t epoch=0;
public:
    bool verify=false;
    uint64_t hit_elements=0,verified_elements=0,published_elements=0;
    StreamRangeCache() {
        const char* flag=std::getenv("POSE_VERIFY_RANGE_CACHE");
        if(flag && std::strcmp(flag,"0") && std::strcmp(flag,"1"))
            throw std::runtime_error("Invalid POSE_VERIFY_RANGE_CACHE");
        verify=flag && !std::strcmp(flag,"1");
    }
    void register_group(int group,int ci,int h,int w,int co,int oh,int ow) {
        if(group<0 || group>=110)throw std::runtime_error("Range cache group");
        const auto& s=STREAM_RANGE_GROUPS[group];
        if(s.ci!=ci || s.h!=h || s.w!=w || s.co!=co || s.oh!=oh || s.ow!=ow)
            throw std::runtime_error("Range cache model shape mismatch");
        if(entries.empty())entries.resize(STREAM_RANGE_SLOT_COUNT);
        registered[group]=true;
    }
    void bind_param(const char* actual) {
        if(std::strcmp(actual,STREAM_RANGE_PARAM))throw std::runtime_error("Range cache graph mismatch");
        for(bool value:registered)if(!value)throw std::runtime_error("Range cache incomplete registration");
        enabled=true;
    }
    void begin_forward() {
        hit_elements=verified_elements=published_elements=0;
        if(++epoch==0) {for(auto& e:entries)e.epoch=0;epoch=1;}
    }
    bool get(int group,int channel,uint64_t fpcr,float& maximum) const {
        if(!enabled || !epoch || group<0 || group>=110)return false;
        const auto& spec=STREAM_RANGE_GROUPS[group];
        if(channel<0 || channel>=spec.ci)return false;
        const int slot=STREAM_RANGE_INPUTS[spec.input_offset+channel];
        if(slot<0)return false;
        const auto& e=entries[slot];
        if(e.epoch!=epoch || e.fpcr!=fpcr)return false;
        maximum=e.maximum;return true;
    }
    bool wants_output(int group,int channel) const {
        return enabled && epoch && group>=0 && group<110 && channel>=0 && channel<STREAM_RANGE_GROUPS[group].co
            && STREAM_RANGE_OUTPUTS[STREAM_RANGE_GROUPS[group].output_offset+channel]>=0;
    }
    void publish(int group,int channel,uint64_t fpcr,float maximum,bool finite) {
        if(!wants_output(group,channel))return;
        const int slot=STREAM_RANGE_OUTPUTS[STREAM_RANGE_GROUPS[group].output_offset+channel];
        auto& e=entries[slot];e.epoch=0;
        if(finite && std::isfinite(maximum) && maximum>=0) {e.maximum=maximum;e.fpcr=fpcr;e.epoch=epoch;}
    }
};
