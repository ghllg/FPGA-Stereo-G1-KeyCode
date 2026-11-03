// ZU5EV transport for the real V06f convolution ABI. Register access stays in kernel.
#pragma once
#include "backend.h"
#include "../hls/protocol.h"
#include "../../board/linux/u02_pose_api.h"
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cstdlib>

struct CoherentBackend : Backend {
    using Clock = std::chrono::steady_clock;
    int fd = -1;
    uint8_t* arena = nullptr;
    size_t capacity = 0, as = 0, bs = 0, cs = 0, bo = 0, co = 0;
    int pm = 0, pn = 0, pk = 0, command_k = 0;
    bool prepared = false, resident = false, failed = false;
    const int8_t *resident_a = nullptr, *resident_image = nullptr;
    int resident_m = 0;
    PoseConv resident_shape = {};
    static size_t aligned(size_t n) { return (n + 63) & ~size_t(63); }
    static double ms(Clock::time_point begin) {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    }
    void release() {
        if (arena) { munmap(arena, capacity); arena = nullptr; }
        if (fd >= 0) { close(fd); fd = -1; }
    }
    explicit CoherentBackend(const char* device_path="/dev/u02_pose",uint32_t abi=U02_POSE_ABI) {
        try {
            fd = open(device_path, O_RDWR | O_CLOEXEC);
            if (fd < 0) throw std::runtime_error(std::string("Cannot open verified ZU5EV ")+device_path);
            u02_pose_info info = {};
            if (ioctl(fd, U02_POSE_INFO, &info) || info.abi != abi ||
                info.arena_bytes != U02_POSE_ARENA_BYTES)
                throw std::runtime_error("ZU5EV kernel transport ABI mismatch");
            capacity = info.arena_bytes;
            void* ptr = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (ptr == MAP_FAILED) throw std::runtime_error("Kernel coherent DMA mmap failed");
            arena = static_cast<uint8_t*>(ptr);
            cached_dma = false; directional_dma = false; fine_profile = true;
        } catch (...) { release(); throw; }
    }
    ~CoherentBackend() override { release(); }
    bool hardware() const override { return true; }
    void copy(void* dst, const void* src, size_t bytes, double& elapsed) {
        auto begin = Clock::now(); memcpy(dst, src, bytes); elapsed += ms(begin);
    }
    void layout(int m, int n, int k, size_t feature_bytes = 0) {
        if (failed || prepared) throw std::runtime_error("Unfinished or failed PL submission; refusing DMA overwrite");
        if (m < 1 || m > POSE_MAX_M || n < 1 || n > POSE_MAX_N || k < 1 || k > POSE_MAX_K)
            throw std::runtime_error("Convolution dimensions exceed V06f contract");
        as = size_t(m) * k; bs = feature_bytes ? feature_bytes : size_t(k) * n;
        cs = size_t(m) * n * 4; bo = aligned(as); co = aligned(bo + bs);
        if (aligned(co + cs) > capacity) throw std::runtime_error("Convolution exceeds coherent DMA arena");
        pm = m; pn = n; pk = command_k = k;
    }
    void execute(Clock::time_point begin,bool legacy=true) {
        u02_pose_job job = {};
        job.a_offset = 0; job.a_bytes = as; job.b_offset = bo; job.b_bytes = bs;
        job.c_offset = co; job.c_bytes = cs; job.m = pm; job.n = pn; job.k = command_k;
        job.timeout_us = 30000000;
        prepare_ms += ms(begin);
        auto submit = Clock::now();
        int rc = ioctl(fd, U02_POSE_RUN, &job);
        double total = ms(submit), hardware_wait = job.elapsed_ns * 1e-6;
        wait_ms += hardware_wait; control_ms += std::max(0.0, total - hardware_wait);
        poll_count += job.polls;
        if (rc) {
            failed = true;
            throw std::runtime_error("ZU5EV kernel submission failed errno=" + std::to_string(errno) +
                                     "; DMA remains kernel-owned until quiescent recovery");
        }
        if (job.status) { failed = true; throw std::runtime_error("V06f PL rejected command: " + std::to_string(job.status)); }
        if(legacy) {++calls; macs += uint64_t(pm) * pn * pk;}
        input_bytes += as + bs; output_bytes += cs;
    }
    int8_t* prepare_input(int m, int n, int k) override {
        const char* flag = std::getenv("POSE_DMA_DIRECT");
        if (!flag || std::string(flag) != "1") return nullptr;
        layout(m, n, k); prepared = true; resident = false;
        return reinterpret_cast<int8_t*>(arena + bo);
    }
    const int32_t* run_prepared(const int8_t* a) override {
        if (!prepared) throw std::runtime_error("No prepared input");
        prepared = false;
        auto begin = Clock::now(); copy(arena, a, as, weight_memcpy_ms); execute(begin);
        ++direct_calls; return reinterpret_cast<const int32_t*>(arena + co);
    }
    void run(const int8_t* a, const int8_t* b, int32_t* c, int m, int n, int k) override {
        layout(m, n, k); resident = false;
        auto begin = Clock::now();
        copy(arena, a, as, weight_memcpy_ms); copy(arena + bo, b, bs, input_memcpy_ms);
        execute(begin); begin = Clock::now();
        copy(c, arena + co, cs, output_memcpy_ms); readback_ms += ms(begin);
    }
    const int32_t* run_convolution(const int8_t* a, const int8_t* image, int32_t*,
        int m, int n, int k, const PoseConv& p, bool first) override {
        if (!pose_conv_valid(p, n, k)) throw std::runtime_error("Invalid direct convolution descriptor");
        uint32_t header[16], previous[16];
        pose_conv_header(header, p); pose_conv_header(previous, resident_shape);
        if (!first && (!resident || resident_a != a || resident_image != image || resident_m != m ||
                       memcmp(header, previous, 48)))
            throw std::runtime_error("Resident convolution image/weight contract changed");
        const size_t image_bytes = size_t(p.w) * p.h * p.channels;
        layout(m, n, k, 64 + image_bytes); command_k = k | POSE_CONV_FLAG;
        auto begin = Clock::now();
        if (first) {
            copy(arena, a, as, weight_memcpy_ms);
            copy(arena + bo + 64, image, image_bytes, input_memcpy_ms);
        }
        copy(arena + bo, header, 64, metadata_memcpy_ms);
        resident = false; execute(begin); resident = true;
        resident_a = a; resident_image = image; resident_m = m; resident_shape = p;
        ++convolution_calls; if (!first) ++resident_reuses;
        return reinterpret_cast<const int32_t*>(arena + co);
    }
    const int32_t* run_depthwise(const int8_t* a, const int8_t* image, int32_t*,
        int channels, int count, int kernel, const PoseConv& p) override {
        if (!light_dw_valid(p, channels, count, kernel)) throw std::runtime_error("Invalid depthwise descriptor");
        const size_t image_bytes = size_t(p.w) * p.h * channels;
        layout(channels, count, kernel, 64 + image_bytes); command_k = kernel | LIGHT_DW_FLAG;
        resident = false; uint32_t header[16]; pose_conv_header(header, p);
        auto begin = Clock::now();
        copy(arena, a, as, weight_memcpy_ms); copy(arena + bo, header, 64, metadata_memcpy_ms);
        copy(arena + bo + 64, image, image_bytes, input_memcpy_ms); execute(begin);
        ++depthwise_calls; return reinterpret_cast<const int32_t*>(arena + co);
    }
};
