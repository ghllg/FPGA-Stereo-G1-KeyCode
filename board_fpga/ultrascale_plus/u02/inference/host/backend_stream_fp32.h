#pragma once
#include "backend_coherent.h"
#include "stream_packet_pool_fp32.h"
#include "../../board/linux/u02_stream_fp32_api.h"
#include <fstream>
#ifdef U02_STREAM_ASYNC_SUBMIT
#include <condition_variable>
#include <mutex>
#include <thread>
#endif

// Synchronous DMA ownership: all framework views stay inside this session's
// kernel allocation. No output guard is filled across neighboring live blobs.
struct StreamFp32Backend : CoherentBackend {
    StreamPacketPoolFp32 pool;
#ifdef U02_STREAM_ASYNC_SUBMIT
    std::mutex async_mutex;
    std::condition_variable async_ready;
    std::thread async_worker;
    bool async_stop=false,async_available=false,async_done=false,async_pending=false;
    uintptr_t async_c_begin=0,async_c_end=0;
    u02_pose_job async_job{};
    int async_rc=0,async_errno=0;
    double async_elapsed=0;
    void worker_loop() {
        std::unique_lock<std::mutex> guard(async_mutex);
        for(;;) {
            async_ready.wait(guard,[&]{return async_stop || async_available;});
            if(async_stop)return;
            u02_pose_job job=async_job;async_available=false;guard.unlock();
            const auto begin=Clock::now();const int rc=ioctl(fd,U02_POSE_RUN,&job);
            const int saved_errno=errno;const double elapsed=ms(begin);
            guard.lock();async_job=job;async_rc=rc;async_errno=saved_errno;
            async_elapsed=elapsed;async_done=true;async_ready.notify_all();
        }
    }
    void finish_forward() override {
        if(!async_pending)return;
        std::unique_lock<std::mutex> guard(async_mutex);
        async_ready.wait(guard,[&]{return async_done;});
        const auto job=async_job;const int rc=async_rc,saved_errno=async_errno;
        const double elapsed=async_elapsed;guard.unlock();async_pending=false;
        const double hardware_wait=job.elapsed_ns*1e-6;
        wait_ms+=hardware_wait;control_ms+=std::max(0.0,elapsed-hardware_wait);poll_count+=job.polls;
        if(rc || job.status) {
            failed=true;
            throw std::runtime_error("Asynchronous FP32 submit failed errno="+std::to_string(saved_errno)+
                " status="+std::to_string(job.status)+"; preserve DMA ownership until quiescent recovery");
        }
    }
    void synchronize_span(const void* memory,size_t bytes) override {
        if(!async_pending || !bytes)return;
        const uintptr_t begin=reinterpret_cast<uintptr_t>(memory);
        if(begin+bytes<begin)throw std::runtime_error("Asynchronous tensor span overflow");
        if(begin<async_c_end && async_c_begin<begin+bytes){++async_span_waits;finish_forward();}
    }
    ~StreamFp32Backend() override {
        // Finish the ioctl before closing/unmapping. Kernel timeout ownership
        // remains pinned; a failed submission is never retried or force-unloaded.
        try{finish_forward();}catch(...){failed=true;}
        {std::lock_guard<std::mutex> guard(async_mutex);async_stop=true;async_ready.notify_all();}
        if(async_worker.joinable())async_worker.join();
    }
#endif
    StreamFp32Backend():CoherentBackend("/dev/u02_stream_fp32",U02_STREAM_FP32_ABI),pool(arena,capacity) {
        // The complete framework arena now shares this mapping. Report the
        // loaded driver's actual cache mode rather than the base default.
        // Mode is read-only for the lifetime of the loaded module.
        std::ifstream parameter("/sys/module/u02_stream_fp32/parameters/cached_dma");
        std::string value;
        if(!(parameter>>value) || (value!="Y" && value!="N" && value!="1" && value!="0"))
            throw std::runtime_error("Cannot verify FP32 driver's DMA cache mode");
        cached_dma=(value=="Y" || value=="1");
#ifdef U02_STREAM_ASYNC_SUBMIT
        async_worker=std::thread([this]{worker_loop();});
#endif
    }
    static void require_fp32_rounding() {
#ifdef __aarch64__
        uint64_t fpcr;asm volatile("mrs %0, fpcr" : "=r"(fpcr));
        // Floating IP implements default round-to-nearest. Do not silently use
        // it under CPU rounding, flush-to-zero or default-NaN modes.
        if(fpcr & ((3ull<<22)|(1ull<<24)|(1ull<<25)))
            throw std::runtime_error("FP32 PL output requires default CPU FP rounding modes");
#endif
    }
    uint8_t* stream_fp32_arena(size_t bytes) override {
        if(failed || prepared || !bytes || bytes>U02_STREAM_FP32_C_POOL_BYTES ||
           U02_STREAM_FP32_C_BASE+bytes>capacity)
            throw std::runtime_error("FP32 framework arena exceeds owned DMA");
        return arena+U02_STREAM_FP32_C_BASE;
    }
    void register_stream_packet(int group,const std::vector<uint64_t>& packet) override {
        if(failed || prepared)throw std::runtime_error("Unfinished FP32 DMA command");
        pool.register_packet(group,packet);
        stream_resident_model_packets=pool.packet_count();
        stream_resident_model_bytes=pool.payload_bytes();
        stream_resident_pool_bytes=pool.used_bytes();
    }
    int8_t* prepare_stream_input(size_t bytes) override {
        if(failed || prepared || !bytes || bytes>U02_STREAM_INPUT_BYTES)
            throw std::runtime_error("Invalid FP32 boundary input");
#ifdef U02_STREAM_ASYNC_SUBMIT
        // The preceding job still owns B. Quantize a ready independent input
        // into this layer's existing private heap plane, then copy after drain.
        if(async_pending){++async_heap_inputs;return nullptr;}
#endif
        ++stream_direct_input_calls;
        return reinterpret_cast<int8_t*>(arena+U02_STREAM_RESIDENT_B_OFFSET);
    }
    void run_stream_fp32_packet(const std::vector<StreamOp>& ops,
            const std::vector<uint64_t>& packet,const int8_t* image,float* output) override {
        if(failed || prepared || ops.empty() || ops.size()>STREAM_MAX_OPS)
            throw std::runtime_error("Invalid or unfinished FP32 submit");
#ifdef U02_STREAM_ASYNC_SUBMIT
        finish_forward();
#endif
        require_fp32_rounding();
        const auto begin=Clock::now();
        const auto& first=ops.front();const auto& last=ops.back();
        const size_t input_bytes_count=size_t(first.ci)*first.h*first.w;
        const size_t output_bytes_count=size_t(last.co)*last.oh*last.ow*4;
        const uintptr_t address=reinterpret_cast<uintptr_t>(output),base=reinterpret_cast<uintptr_t>(arena);
        if(address<base+U02_STREAM_FP32_C_BASE || address-base>=U02_STREAM_FP32_C_BASE+U02_STREAM_FP32_C_POOL_BYTES ||
           (address-base)%64 || !output_bytes_count || output_bytes_count%64 ||
           output_bytes_count>U02_STREAM_OUTPUT_BYTES || address-base+output_bytes_count>U02_STREAM_FP32_C_BASE+U02_STREAM_FP32_C_POOL_BYTES ||
           !input_bytes_count || input_bytes_count>U02_STREAM_INPUT_BYTES)
            throw std::runtime_error("FP32 output is not a whole aligned owned tensor");
        const auto copy_begin=Clock::now();const auto slot=pool.update_prefix(packet);
        weight_memcpy_ms+=ms(copy_begin);
        if(image!=reinterpret_cast<const int8_t*>(arena+U02_STREAM_RESIDENT_B_OFFSET))
            copy(arena+U02_STREAM_RESIDENT_B_OFFSET,image,input_bytes_count,input_memcpy_ms);
        u02_pose_job job={};
        job.a_offset=slot.offset;job.a_bytes=job.n=slot.bytes;
        job.b_offset=U02_STREAM_RESIDENT_B_OFFSET;job.b_bytes=input_bytes_count;
        job.c_offset=address-base;job.c_bytes=output_bytes_count;
        job.m=ops.size();job.k=U02_STREAM_MAGIC;job.timeout_us=30000000;
        prepare_ms+=ms(begin);
#ifdef U02_STREAM_ASYNC_SUBMIT
        {
            std::lock_guard<std::mutex> guard(async_mutex);
            async_job=job;async_done=false;async_available=true;
            // Cover the 64-byte guard AND cache-maintenance rounding beyond
            // it. Conservative 256-byte exclusion also covers ARM64's larger
            // DMA minimum alignment; neighboring tensors must stay untouched.
            async_c_begin=address;async_c_end=address+output_bytes_count+256;
            async_pending=true;++async_submissions;async_ready.notify_all();
        }
#else
        const auto submit=Clock::now();const int rc=ioctl(fd,U02_POSE_RUN,&job);
        const double elapsed=ms(submit),hardware_wait=job.elapsed_ns*1e-6;
        wait_ms+=hardware_wait;control_ms+=std::max(0.0,elapsed-hardware_wait);poll_count+=job.polls;
        if(rc || job.status) {
            failed=true;
            throw std::runtime_error("FP32 submission failed errno="+std::to_string(errno)+" status="+
                std::to_string(job.status)+"; retain DMA ownership until quiescent recovery");
        }
#endif
        ++calls;++stream_calls;++stream_prepared_packet_calls;++stream_resident_packet_calls;
        stream_resident_prefix_copy_bytes+=slot.prefix;
        input_bytes+=slot.bytes+input_bytes_count;output_bytes+=output_bytes_count;
        for(const auto& op:ops) {
            if(op.kind)++depthwise_calls;else ++convolution_calls;
            macs+=uint64_t(op.co)*op.oh*op.ow*(op.kind?9:op.ci);
        }
    }
};
