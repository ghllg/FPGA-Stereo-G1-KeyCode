#pragma once
#include "backend_coherent.h"
#include "stream_wire.h"
#include "../../board/linux/u02_stream_api.h"

struct StreamCoherentBackend : CoherentBackend {
    StreamCoherentBackend():CoherentBackend("/dev/u02_stream",U02_STREAM_ABI) {}
    int8_t* prepare_stream_input(size_t bytes) override {
        if(failed || prepared)throw std::runtime_error("Unfinished streaming DMA command");
        if(!bytes || bytes>U02_STREAM_INPUT_BYTES || U02_STREAM_B_OFFSET+bytes>capacity)
            throw std::runtime_error("Streaming input exceeds fixed DMA slot");
        ++stream_direct_input_calls;
        return reinterpret_cast<int8_t*>(arena+U02_STREAM_B_OFFSET);
    }
    const int32_t* run_stream(const std::vector<StreamOp>& ops,const std::vector<int8_t>& weights,
                    const std::vector<StreamRequant>& coeff,const int8_t* image,int32_t*) override {
        if(failed || prepared)throw std::runtime_error("Unfinished streaming DMA command");
        const auto begin=Clock::now();
        const auto packet=stream_pack(ops,weights,coeff);
        return execute_packet(ops,packet,image,begin);
    }
    const int32_t* run_stream_packet(const std::vector<StreamOp>& ops,
                    const std::vector<uint64_t>& packet,const int8_t* image) override {
        if(failed || prepared)throw std::runtime_error("Unfinished streaming DMA command");
        const int32_t* result=execute_packet(ops,packet,image,Clock::now());
        ++stream_prepared_packet_calls;
        return result;
    }
private:
    const int32_t* execute_packet(const std::vector<StreamOp>& ops,
                    const std::vector<uint64_t>& packet,const int8_t* image,Clock::time_point begin) {
        if(ops.empty() || ops.size()>STREAM_MAX_OPS)
            throw std::runtime_error("Streaming operator count");
        const auto& first=ops.front();const auto& last=ops.back();
        as=packet.size()*8;bs=size_t(first.ci)*first.h*first.w;cs=size_t(last.co)*last.oh*last.ow*4;
        bo=U02_STREAM_B_OFFSET;co=U02_STREAM_C_OFFSET;
        if(as>U02_STREAM_PACKET_BYTES || bs>U02_STREAM_INPUT_BYTES || cs>U02_STREAM_OUTPUT_BYTES ||
           co+U02_STREAM_OUTPUT_BYTES>capacity)throw std::runtime_error("Streaming packet exceeds fixed DMA slots");
        pm=ops.size();pn=as;pk=0;command_k=U02_STREAM_MAGIC;
        copy(arena,packet.data(),as,weight_memcpy_ms);
        if(image!=reinterpret_cast<const int8_t*>(arena+bo))
            copy(arena+bo,image,bs,input_memcpy_ms);
        execute(begin,false);
        ++calls;++stream_calls;
        for(const auto& op:ops) {
            if(op.kind)++depthwise_calls;else ++convolution_calls;
            macs+=uint64_t(op.co)*op.oh*op.ow*(op.kind?9:op.ci);
        }
        // Successful synchronous ioctl includes the driver's completion and
        // DMA read barrier. Dequantize before the next submission overwrites C.
        return reinterpret_cast<const int32_t*>(arena+co);
    }
};
