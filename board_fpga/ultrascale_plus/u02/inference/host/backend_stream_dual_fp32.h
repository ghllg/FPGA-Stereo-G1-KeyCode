#pragma once
#include "backend_stream_dual.h"
#include "stream_packet_pool_fp32.h"
#include "../../board/linux/u02_stream_dual_fp32_api.h"
#include <fstream>

// Reuse the serialized shared FD, failure poisoning and private context lease.
// Each session has its own FP32 framework arena and packet/quantized input.
struct StreamDualFp32Backend : StreamDualBackend {
    StreamPacketPoolFp32 fp32_pool;
    StreamDualFp32Backend(std::shared_ptr<StreamDualDevice> owner,unsigned ctx):
        StreamDualBackend(owner,ctx),fp32_pool(owner->memory()+U02_STREAM_DUAL_A_OFFSET(ctx),
            U02_STREAM_DUAL_FP32_ARENA_BYTES-U02_STREAM_DUAL_A_OFFSET(ctx)) {
        if(owner->capacity()!=U02_STREAM_DUAL_FP32_ARENA_BYTES || owner->abi()!=U02_STREAM_DUAL_FP32_ABI)
            throw std::runtime_error("FP32 private arenas require the owned ABI8 mapping");
        std::ifstream parameter("/sys/module/u02_stream_dual_fp32/parameters/cached_dma");
        std::string value;
        if(!(parameter>>value) || (value!="Y" && value!="N" && value!="1" && value!="0"))
            throw std::runtime_error("Cannot verify dual FP32 DMA cache mode");
        cached_dma=(value=="Y" || value=="1");
    }
    uint8_t* stream_fp32_arena(size_t bytes) override {
        device->check();
        if(!bytes || bytes>U02_STREAM_DUAL_FP32_C_POOL_BYTES)throw std::runtime_error("Private FP32 framework arena capacity");
        return device->memory()+U02_STREAM_DUAL_FP32_C_OFFSET(context);
    }
    void register_stream_packet(int group,const std::vector<uint64_t>& packet) override {
        device->check();fp32_pool.register_packet(group,packet);
        stream_resident_model_packets=fp32_pool.packet_count();stream_resident_model_bytes=fp32_pool.payload_bytes();
        stream_resident_pool_bytes=fp32_pool.used_bytes();
    }
    const int32_t* run_stream_packet(const std::vector<StreamOp>&,
            const std::vector<uint64_t>&,const int8_t*) override {
        throw std::runtime_error("FP32 context refuses integer output submissions");
    }
    void run_stream_fp32_packet(const std::vector<StreamOp>& ops,
            const std::vector<uint64_t>& packet,const int8_t* image,float* output) override {
        device->check();const auto begin=Clock::now();
#ifdef __aarch64__
        uint64_t fpcr;asm volatile("mrs %0, fpcr":"=r"(fpcr));
        if(fpcr&((3ull<<22)|(1ull<<24)|(1ull<<25)))throw std::runtime_error("FP32 hardware requires default FPCR");
#endif
        if(ops.empty() || ops.size()>STREAM_MAX_OPS)throw std::runtime_error("FP32 operator count");
        const auto& first=ops.front();const auto& last=ops.back();
        const size_t input=size_t(first.ci)*first.h*first.w,bytes=size_t(last.co)*last.oh*last.ow*4;
        const uintptr_t address=reinterpret_cast<uintptr_t>(output),base=reinterpret_cast<uintptr_t>(device->memory());
        const size_t cbase=U02_STREAM_DUAL_FP32_C_OFFSET(context);
        if(address<base+cbase || address-base>=cbase+U02_STREAM_DUAL_FP32_C_POOL_BYTES ||
           (address-base)%64 || !bytes || bytes%64 || bytes>U02_STREAM_OUTPUT_BYTES ||
           address-base+bytes>cbase+U02_STREAM_DUAL_FP32_C_POOL_BYTES || !input || input>U02_STREAM_INPUT_BYTES)
            throw std::runtime_error("FP32 output must be a complete aligned private tensor");
        auto started=Clock::now();const auto slot=fp32_pool.update_prefix(packet);
        weight_memcpy_ms+=StreamDualDevice::ms(started);
        const unsigned a=U02_STREAM_DUAL_A_OFFSET(context),b=U02_STREAM_DUAL_B_OFFSET(context);
        if(image!=reinterpret_cast<const int8_t*>(device->memory()+b)) {
            started=Clock::now();memcpy(device->memory()+b,image,input);input_memcpy_ms+=StreamDualDevice::ms(started);
        }
        u02_pose_job job={};job.a_offset=a+slot.offset;job.a_bytes=job.n=slot.bytes;
        job.b_offset=b;job.b_bytes=input;job.c_offset=address-base;job.c_bytes=bytes;
        job.m=ops.size();job.k=U02_STREAM_MAGIC;job.timeout_us=30000000;
        if(!u02_stream_dual_fp32_slots_valid(&job))throw std::runtime_error("FP32 context mismatch before submit");
        // No sentinel fill: neighboring live framework tensors share C. The
        // kernel still transfers the full guarded cache ownership envelope.
        prepare_ms+=StreamDualDevice::ms(begin);
        const double old_queue=queue_wait_ms;double ioctl_ms=0;device->run(job,queue_wait_ms,ioctl_ms);
        const double waited=job.elapsed_ns*1e-6;
        wait_ms+=waited;control_ms+=std::max(0.0,ioctl_ms-waited)+(queue_wait_ms-old_queue);poll_count+=job.polls;
        ++calls;++stream_calls;++stream_prepared_packet_calls;++stream_resident_packet_calls;
        stream_resident_prefix_copy_bytes+=slot.prefix;input_bytes+=slot.bytes+input;output_bytes+=bytes;
        for(const auto& op:ops) {if(op.kind)++depthwise_calls;else ++convolution_calls;
            macs+=uint64_t(op.co)*op.oh*op.ow*(op.kind?9:op.ci);}
    }
};
