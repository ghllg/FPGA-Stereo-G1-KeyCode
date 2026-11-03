#pragma once
#include "backend_stream_dual.h"
#include "stream_shared_packet_pool.h"

// Only this wrapper can reach the underlying transport. Its outer lock covers
// shared A publication AND the synchronous ioctl, including timeout poisoning.
class StreamPipelineDevice {
    StreamDualDevice transport;
    StreamSharedPacketPool pool;
    std::mutex submission,leases;
    std::array<bool,4> leased{{false,false,false,false}};
    unsigned contexts;
public:
    explicit StreamPipelineDevice(unsigned count):
        transport("/dev/u02_stream_pipeline",U02_STREAM_PIPELINE_ABI),
        pool(transport.memory(),transport.capacity(),count),contexts(count) {}
    uint8_t* memory() const {return transport.memory();}
    void check() const {transport.check();}
    void poison() {transport.poison();}
    void claim(unsigned ctx) {
        std::lock_guard<std::mutex> lock(leases);check();
        if(ctx>=contexts || leased[ctx])throw std::runtime_error("Pipeline context ownership");
        leased[ctx]=true;
    }
    void unclaim(unsigned ctx) {std::lock_guard<std::mutex> lock(leases);leased[ctx]=false;}
    void register_packet(unsigned ctx,int group,const std::vector<uint64_t>& packet) {
        pool.register_packet(ctx,group,packet);
    }
    void seal() {pool.seal();}
    size_t model_bytes() const {return pool.payload_bytes();}
    size_t pool_bytes() const {return pool.used_bytes();}
    size_t packet_count() const {return pool.packet_count();}
    void run(unsigned ctx,const std::vector<uint64_t>& packet,u02_pose_job& job,
             double& queue_ms,double& prefix_ms,double& ioctl_ms,uint64_t& prefix_bytes
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
             ,U02IoctlCpuProfile* cpu_profile=nullptr
#endif
             ) {
        const auto waiting=StreamDualDevice::Clock::now();
        std::lock_guard<std::mutex> lock(submission);
        queue_ms+=StreamDualDevice::ms(waiting);check();
        const auto begin=StreamDualDevice::Clock::now();
        const auto slot=pool.update_prefix(ctx,packet);
        prefix_ms+=StreamDualDevice::ms(begin);prefix_bytes+=slot.prefix;
        job.a_offset=slot.offset;job.a_bytes=job.n=slot.bytes;
        // Inner transport also validates the device result and poisons failure.
        double inner_queue=0;
        transport.run(job,inner_queue,ioctl_ms
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
                      ,cpu_profile
#endif
                      );
        queue_ms+=inner_queue;
    }
};

struct StreamPipelineBackend : Backend {
    std::shared_ptr<StreamPipelineDevice> device;
    unsigned context;
    double queue_wait_ms=0;
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
    U02IoctlCpuProfile ioctl_cpu;
#endif
    StreamPipelineBackend(std::shared_ptr<StreamPipelineDevice> owner,unsigned ctx):device(std::move(owner)),context(ctx) {
        device->claim(ctx);fine_profile=true;
    }
    ~StreamPipelineBackend() override {device->unclaim(context);}
    bool hardware() const override {return true;}
    void run(const int8_t*,const int8_t*,int32_t*,int,int,int) override {
        throw std::runtime_error("Pipeline backend requires registered streaming packets");
    }
    void register_stream_packet(int group,const std::vector<uint64_t>& packet) override {
        device->check();device->register_packet(context,group,packet);
        stream_resident_model_packets=device->packet_count();stream_resident_model_bytes=device->model_bytes();
        stream_resident_pool_bytes=device->pool_bytes();
    }
    int8_t* prepare_stream_input(size_t bytes) override {
        device->check();
        if(!bytes || bytes>U02_STREAM_INPUT_BYTES)throw std::runtime_error("Pipeline input capacity");
        ++stream_direct_input_calls;
        return reinterpret_cast<int8_t*>(device->memory()+U02_STREAM_PIPELINE_B_OFFSET(context));
    }
    const int32_t* run_stream_packet(const std::vector<StreamOp>& ops,
            const std::vector<uint64_t>& packet,const int8_t* image) override {
        device->check();const auto begin=StreamDualDevice::Clock::now();
        if(ops.empty() || ops.size()>STREAM_MAX_OPS)throw std::runtime_error("Pipeline operator count");
        const auto& first=ops.front();const auto& last=ops.back();
        const size_t input=size_t(first.ci)*first.h*first.w,output=size_t(last.co)*last.oh*last.ow*4;
        if(!input || input>U02_STREAM_INPUT_BYTES || !output || output>U02_STREAM_OUTPUT_BYTES)
            throw std::runtime_error("Pipeline tensor capacity");
        uint8_t* arena=device->memory();const unsigned bo=U02_STREAM_PIPELINE_B_OFFSET(context),co=U02_STREAM_PIPELINE_C_OFFSET(context);
        if(image!=reinterpret_cast<const int8_t*>(arena+bo)) {
            const auto copy=StreamDualDevice::Clock::now();memcpy(arena+bo,image,input);input_memcpy_ms+=StreamDualDevice::ms(copy);
        }
        memset(arena+co+output,0xa5,64);
        u02_pose_job job={};job.b_offset=bo;job.b_bytes=input;job.c_offset=co;job.c_bytes=output;
        job.m=ops.size();job.k=U02_STREAM_MAGIC;job.timeout_us=30000000;
        prepare_ms+=StreamDualDevice::ms(begin);
        const double old_queue=queue_wait_ms;double ioctl_ms=0;
        device->run(context,packet,job,queue_wait_ms,weight_memcpy_ms,ioctl_ms,stream_resident_prefix_copy_bytes
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
                    ,&ioctl_cpu
#endif
                    );
        const double waited=job.elapsed_ns*1e-6;
        wait_ms+=waited;control_ms+=std::max(0.0,ioctl_ms-waited)+(queue_wait_ms-old_queue);poll_count+=job.polls;
        for(unsigned i=0;i<64;++i)if(arena[co+output+i]!=0xa5) {
            device->poison();throw std::runtime_error("Pipeline output guard overwritten; preserve DMA ownership");
        }
        ++calls;++stream_calls;++stream_prepared_packet_calls;++stream_resident_packet_calls;
        input_bytes+=job.a_bytes+input;output_bytes+=output;
        for(const auto& op:ops) {
            if(op.kind)++depthwise_calls;else ++convolution_calls;
            macs+=uint64_t(op.co)*op.oh*op.ow*(op.kind?9:op.ci);
        }
        return reinterpret_cast<const int32_t*>(arena+co);
    }
};
