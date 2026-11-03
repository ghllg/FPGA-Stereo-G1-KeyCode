#pragma once
#include <time.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

// This clock advances only while the calling thread consumes CPU. It includes
// that thread's user and kernel execution, not mutex/scheduler sleep or other
// threads' work. Counter-read overhead is deliberately not subtracted.
inline uint64_t u02_thread_cpu_ns() {
    timespec value{};
    if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&value) || value.tv_sec<0 || value.tv_nsec<0 || value.tv_nsec>=1000000000)
        throw std::runtime_error("Calling-thread CPU clock unavailable");
    return uint64_t(value.tv_sec)*1000000000ull+uint64_t(value.tv_nsec);
}
struct U02IoctlCpuSample {
    uint64_t cpu_ns,wall_ns,driver_ns,polls;
    uint32_t a_bytes,b_bytes,c_bytes,ops;
};
struct U02IoctlCpuProfile {
    uint64_t cpu_ns=0;
    std::vector<U02IoctlCpuSample> rows;
    U02IoctlCpuProfile(){rows.reserve(110);}
    void reset(){cpu_ns=0;rows.clear();}
    void add(const U02IoctlCpuSample& sample){cpu_ns+=sample.cpu_ns;rows.push_back(sample);}
};
