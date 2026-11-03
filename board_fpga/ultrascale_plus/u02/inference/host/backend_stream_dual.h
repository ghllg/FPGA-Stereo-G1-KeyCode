#pragma once
#include "backend.h"
#include "stream_packet_pool.h"
#include "../../board/linux/u02_stream_dual_api.h"
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cerrno>
#include <algorithm>
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
#include "thread_cpu_profile.h"
#endif

// One owned FD/mapping for both contexts, one serialized hardware command.
// CPU work in a context never borrows the other context's packet/input/output.
class StreamDualDevice {
    int fd=-1;
    uint8_t* mapping=nullptr;
    size_t mapping_bytes=U02_POSE_ARENA_BYTES;
    unsigned device_abi=0;
    std::mutex submission,lease_lock;
    bool leased[2]={false,false};
    std::atomic<bool> failed{false};
    void release() {
        // Closing a failed owner does not release the kernel's inflight module
        // reference/allocation. Recovery is external and requires actual idle.
        if(mapping) {munmap(mapping,mapping_bytes);mapping=nullptr;}
        if(fd>=0) {close(fd);fd=-1;}
    }
public:
    using Clock=std::chrono::steady_clock;
    static double ms(Clock::time_point start) {
        return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    }
    StreamDualDevice(const char* path="/dev/u02_stream_dual",unsigned abi=U02_STREAM_DUAL_ABI,
                     size_t bytes=U02_POSE_ARENA_BYTES):mapping_bytes(bytes) {
        try {
            if(!bytes || bytes%4096)throw std::runtime_error("Invalid owned DMA capacity");
            fd=open(path,O_RDWR|O_CLOEXEC);
            if(fd<0) throw std::runtime_error("Cannot open verified ABI7 dual-context device");
            u02_pose_info info={};
            if(ioctl(fd,U02_POSE_INFO,&info) || info.abi!=abi || info.arena_bytes!=bytes)
                throw std::runtime_error("Dual-context kernel ABI/capacity mismatch");
            device_abi=info.abi;
            void* ptr=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
            if(ptr==MAP_FAILED)throw std::runtime_error("Dual-context DMA mapping failed");
            mapping=static_cast<uint8_t*>(ptr);
        } catch(...) {release();throw;}
    }
    ~StreamDualDevice() {release();}
    StreamDualDevice(const StreamDualDevice&)=delete;
    StreamDualDevice& operator=(const StreamDualDevice&)=delete;
    void check() const {
        if(failed.load(std::memory_order_acquire))
            throw std::runtime_error("Shared PL device failed; preserve DMA until independent idle recovery");
    }
    uint8_t* memory() const {return mapping;}
    size_t capacity() const {return mapping_bytes;}
    unsigned abi() const {return device_abi;}
    void poison() {failed.store(true,std::memory_order_release);}
    void claim(unsigned ctx) {
        std::lock_guard<std::mutex> guard(lease_lock);check();
        if(ctx>=2 || leased[ctx])throw std::runtime_error("Dual context already owned or invalid");
        leased[ctx]=true;
    }
    void unclaim(unsigned ctx) {std::lock_guard<std::mutex> guard(lease_lock);leased[ctx]=false;}
    void run(u02_pose_job& job,double& queue_ms,double& ioctl_ms
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
             ,U02IoctlCpuProfile* cpu_profile=nullptr
#endif
             ) {
        const auto waiting=Clock::now();
        std::lock_guard<std::mutex> guard(submission);
        queue_ms+=ms(waiting);check();
        const auto start=Clock::now();
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
        uint64_t cpu_start=0;
        try {if(cpu_profile)cpu_start=u02_thread_cpu_ns();}
        catch(...){poison();throw;}
#endif
        const int rc=ioctl(fd,U02_POSE_RUN,&job),saved_errno=errno;
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
        // Poison before another diagnostic operation can throw after a failed
        // command. Kernel inflight ownership and external recovery stay intact.
        if(rc || job.status)poison();
        try {
            const auto cpu_end=cpu_profile?u02_thread_cpu_ns():0;
            const auto end=Clock::now();
            ioctl_ms=std::chrono::duration<double,std::milli>(end-start).count();
            if(cpu_profile) {
                if(cpu_end<cpu_start)throw std::runtime_error("Thread CPU clock reversed");
                cpu_profile->add({cpu_end-cpu_start,uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()),
                    job.elapsed_ns,job.polls,job.a_bytes,job.b_bytes,job.c_bytes,job.m});
            }
        }catch(...){poison();throw;}
#else
        ioctl_ms=ms(start);
#endif
        if(rc || job.status) {
            failed.store(true,std::memory_order_release);
            throw std::runtime_error("Dual-context submission failed errno="+std::to_string(saved_errno)+
                " status="+std::to_string(job.status)+"; do not unload/reprogram without idle proof");
        }
    }
};

// Exactly one model/worker owns a context. Both workers share only Device.
struct StreamDualBackend : Backend {
    using Clock=StreamDualDevice::Clock;
    std::shared_ptr<StreamDualDevice> device;
    unsigned context;
    StreamPacketPool pool;
    double queue_wait_ms=0;
    StreamDualBackend(std::shared_ptr<StreamDualDevice> owner,unsigned ctx):device(std::move(owner)),
        context(ctx),pool(device->memory(),U02_POSE_ARENA_BYTES,U02_STREAM_DUAL_A_OFFSET(ctx)) {
        device->claim(ctx);fine_profile=true;
    }
    ~StreamDualBackend() override {device->unclaim(context);}
    bool hardware() const override {return true;}
    void run(const int8_t*,const int8_t*,int32_t*,int,int,int) override {
        throw std::runtime_error("Dual context accepts registered streaming packets only");
    }
    void register_stream_packet(int group,const std::vector<uint64_t>& packet) override {
        device->check();pool.register_packet(group,packet);
        stream_resident_model_packets=pool.packet_count();stream_resident_model_bytes=pool.payload_bytes();
        stream_resident_pool_bytes=pool.used_bytes();
    }
    int8_t* prepare_stream_input(size_t bytes) override {
        device->check();
        if(!bytes || bytes>U02_STREAM_INPUT_BYTES)throw std::runtime_error("Dual-context input capacity");
        ++stream_direct_input_calls;
        return reinterpret_cast<int8_t*>(device->memory()+U02_STREAM_DUAL_B_OFFSET(context));
    }
    const int32_t* run_stream_packet(const std::vector<StreamOp>& ops,
            const std::vector<uint64_t>& packet,const int8_t* image) override {
        device->check();const auto begin=Clock::now();
        if(ops.empty() || ops.size()>STREAM_MAX_OPS)throw std::runtime_error("Dual-context operator count");
        const auto& first=ops.front();const auto& last=ops.back();
        const size_t input=size_t(first.ci)*first.h*first.w,output=size_t(last.co)*last.oh*last.ow*4;
        if(!input || input>U02_STREAM_INPUT_BYTES || !output || output>U02_STREAM_OUTPUT_BYTES)
            throw std::runtime_error("Dual-context tensor capacity");
        auto started=Clock::now();const auto slot=pool.update_prefix(packet);
        weight_memcpy_ms+=StreamDualDevice::ms(started);
        uint8_t* arena=device->memory();const unsigned bo=U02_STREAM_DUAL_B_OFFSET(context),co=U02_STREAM_DUAL_C_OFFSET(context);
        if(image!=reinterpret_cast<const int8_t*>(arena+bo)) {
            started=Clock::now();memcpy(arena+bo,image,input);input_memcpy_ms+=StreamDualDevice::ms(started);
        }
        // Output bytes plus guard are owned only by this context and only
        // consumed after synchronous completion; check after every command.
        memset(arena+co+output,0xa5,64);
        u02_pose_job job={};job.a_offset=slot.offset;job.a_bytes=job.n=slot.bytes;
        job.b_offset=bo;job.b_bytes=input;job.c_offset=co;job.c_bytes=output;
        job.m=ops.size();job.k=U02_STREAM_MAGIC;job.timeout_us=30000000;
        prepare_ms+=StreamDualDevice::ms(begin);
        const double old_queue=queue_wait_ms;double ioctl_ms=0;
        device->run(job,queue_wait_ms,ioctl_ms);
        const double waited=job.elapsed_ns*1e-6;
        wait_ms+=waited;control_ms+=std::max(0.0,ioctl_ms-waited)+(queue_wait_ms-old_queue);poll_count+=job.polls;
        for(unsigned i=0;i<64;++i) if(arena[co+output+i]!=0xa5) {
            device->poison();
            throw std::runtime_error("Dual-context output guard overwritten; stop candidate");
        }
        ++calls;++stream_calls;++stream_prepared_packet_calls;++stream_resident_packet_calls;
        stream_resident_prefix_copy_bytes+=slot.prefix;input_bytes+=slot.bytes+input;output_bytes+=output;
        for(const auto& op:ops) {
            if(op.kind)++depthwise_calls;else ++convolution_calls;
            macs+=uint64_t(op.co)*op.oh*op.ow*(op.kind?9:op.ci);
        }
        return reinterpret_cast<const int32_t*>(arena+co);
    }
};
