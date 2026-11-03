#pragma once
#include "backend_coherent.h"
#include "stream_packet_pool.h"

struct StreamResidentBackend : CoherentBackend {
    StreamPacketPool pool;
    StreamResidentBackend():CoherentBackend("/dev/u02_stream_resident",U02_STREAM_RESIDENT_ABI),pool(arena,capacity) {}
    void register_stream_packet(int group,const std::vector<uint64_t>& packet) override {
        if(failed || prepared)throw std::runtime_error("Unfinished resident DMA command");
        pool.register_packet(group,packet);
        stream_resident_model_packets=pool.packet_count();
        stream_resident_model_bytes=pool.payload_bytes();
        stream_resident_pool_bytes=pool.used_bytes();
    }
    int8_t* prepare_stream_input(size_t bytes) override {
        if(failed || prepared)throw std::runtime_error("Unfinished resident DMA command");
        if(!bytes || bytes>U02_STREAM_INPUT_BYTES || U02_STREAM_RESIDENT_B_OFFSET+bytes>capacity)
            throw std::runtime_error("Resident input exceeds fixed DMA slot");
        ++stream_direct_input_calls;
        return reinterpret_cast<int8_t*>(arena+U02_STREAM_RESIDENT_B_OFFSET);
    }
    const int32_t* run_stream_packet(const std::vector<StreamOp>& ops,
            const std::vector<uint64_t>& packet,const int8_t* image) override {
        if(failed || prepared)throw std::runtime_error("Unfinished resident DMA command");
        if(ops.empty() || ops.size()>STREAM_MAX_OPS)throw std::runtime_error("Resident operator count");
        const auto begin=Clock::now();
        const auto& first=ops.front();const auto& last=ops.back();
        const size_t input=size_t(first.ci)*first.h*first.w;
        const size_t output=size_t(last.co)*last.oh*last.ow*4;
        if(!input || input>U02_STREAM_INPUT_BYTES || !output || output>U02_STREAM_OUTPUT_BYTES)
            throw std::runtime_error("Resident tensor exceeds fixed DMA slot");
        const auto copy_begin=Clock::now();
        const auto slot=pool.update_prefix(packet);
        weight_memcpy_ms+=ms(copy_begin);
        if(image!=reinterpret_cast<const int8_t*>(arena+U02_STREAM_RESIDENT_B_OFFSET))
            copy(arena+U02_STREAM_RESIDENT_B_OFFSET,image,input,input_memcpy_ms);
        u02_pose_job job={};
        job.a_offset=slot.offset;job.a_bytes=job.n=slot.bytes;
        job.b_offset=U02_STREAM_RESIDENT_B_OFFSET;job.b_bytes=input;
        job.c_offset=U02_STREAM_RESIDENT_C_OFFSET;job.c_bytes=output;
        job.m=ops.size();job.k=U02_STREAM_MAGIC;job.timeout_us=30000000;
        prepare_ms+=ms(begin);
        const auto submit=Clock::now();
        const int rc=ioctl(fd,U02_POSE_RUN,&job);
        const double elapsed=ms(submit),hardware_wait=job.elapsed_ns*1e-6;
        wait_ms+=hardware_wait;control_ms+=std::max(0.0,elapsed-hardware_wait);poll_count+=job.polls;
#ifdef U02_CALL_DIAGNOSTIC
        diagnose("resident_packet",ops,elapsed,hardware_wait,job.polls);
#endif
        if(rc || job.status) {
            failed=true;
            throw std::runtime_error("Resident submission failed errno="+std::to_string(errno)+
                " status="+std::to_string(job.status)+"; preserve kernel-owned DMA until quiescent recovery");
        }
        ++calls;++stream_calls;++stream_prepared_packet_calls;++stream_resident_packet_calls;
        stream_resident_prefix_copy_bytes+=slot.prefix;
        input_bytes+=slot.bytes+input;output_bytes+=output;
        for(const auto& op:ops) {
            if(op.kind)++depthwise_calls;else ++convolution_calls;
            macs+=uint64_t(op.co)*op.oh*op.ow*(op.kind?9:op.ci);
        }
        return reinterpret_cast<const int32_t*>(arena+U02_STREAM_RESIDENT_C_OFFSET);
    }
};
