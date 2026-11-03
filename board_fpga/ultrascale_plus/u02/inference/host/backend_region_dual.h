#pragma once
#include "backend.h"
#include "../../board/linux/u02_region_dual_api.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstddef>
#include <vector>
#ifdef U02_STREAM_REGION_HOOK
#include "region_model_binding.h"
#include "region_graph_dispatch.h"
#endif

#ifndef U02_STREAM_FP32_OUTPUT
#error Dual region backend requires packed FP32 streaming output
#endif
// Ordinary groups for the candidate shared engine. Framework tensors are
// CPU-owned aligned storage; the driver copies packet/input into private DMA
// and this transport copies completed scratch output back into the framework.
// It does not claim resident packets, direct DMA output or reduced copy cost.
template<class Transport> class RegionDualBackend : public Backend {
    Transport transport_;
    std::vector<unsigned char> storage_;
    unsigned char* base_=nullptr;
    size_t capacity_=0;
#ifdef U02_STREAM_REGION_HOOK
    RegionModelBinding binding_;
    std::unique_ptr<RegionGraphDispatch<RegionDualBackend<Transport>>> extension_;
    uint64_t region_macs_=0,region_conv_=0,region_dw_=0;
#ifdef U02_CALL_DIAGNOSTIC
    std::vector<StreamOp> diagnostic_region_ops_;
#endif
#endif
public:
#ifdef U02_STREAM_REGION_HOOK
    explicit RegionDualBackend(const std::vector<unsigned char>& constants={}):binding_(constants) {}
    void register_stream_packet(int id,const std::vector<uint64_t>& packet) override {
        if(id<0)throw std::runtime_error("Invalid region group ID");binding_.packet(unsigned(id),packet);
    }
    void register_stream_boundary(int id,const std::vector<float>& weight,const std::vector<float>& bias,
                                  const std::vector<float>& outgoing) override {
        if(id<0)throw std::runtime_error("Invalid region group ID");binding_.boundary(unsigned(id),weight,bias,outgoing);
    }
    RegionScheduleExtension* region_extension() override {
        if(!extension_) {
            const auto& model=binding_.seal();RegionModelHeader h;memcpy(&h,model.data(),64);
            for(unsigned i=0;i<h.groups;++i) {
                RegionModelGroup g;memcpy(&g,model.data()+64+i*64,64);
                uint32_t p[8];memcpy(p,model.data()+g.packet_offset,32);
                for(unsigned j=0;j<p[2];++j) {
                    StreamOp op;memcpy(&op,model.data()+g.packet_offset+32+j*48,48);
#ifdef U02_CALL_DIAGNOSTIC
                    diagnostic_region_ops_.push_back(op);
#endif
                    region_macs_+=uint64_t(op.co)*op.oh*op.ow*(op.kind?9:op.ci);
                    if(op.kind)++region_dw_;else ++region_conv_;
                }
            }
            extension_.reset(new RegionGraphDispatch<RegionDualBackend<Transport>>(*this,model));
        }
        return extension_.get();
    }
    void install(const std::vector<unsigned char>& metadata,const std::vector<unsigned char>& model) {
        if(model!=binding_.seal())throw std::runtime_error("Fused model changed after binding");
        transport_.install(metadata,model);
    }
    void write_ingress(unsigned id,const float* input,size_t elements){transport_.write_ingress(id,input,elements);}
    void read_egress(unsigned id,float* output,size_t elements){transport_.read_egress(id,output,elements);}
    u02_region_job run() {
        const auto begin=std::chrono::steady_clock::now();const auto job=transport_.run(30000000);
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        wait_ms+=job.elapsed_ns*1e-6;control_ms+=std::max(0.0,elapsed-job.elapsed_ns*1e-6);poll_count+=job.polls;
#ifdef U02_CALL_DIAGNOSTIC
        diagnose("fused_region",diagnostic_region_ops_,elapsed,job.elapsed_ns*1e-6,job.polls);
#endif
        ++calls;++stream_calls;macs+=region_macs_;convolution_calls+=region_conv_;depthwise_calls+=region_dw_;
        return job;
    }
#endif
    static void require_fp32_rounding() {
#if defined(__aarch64__)
        uint64_t fpcr;asm volatile("mrs %0, fpcr":"=r"(fpcr));
        if(fpcr & ((uint64_t(3)<<22)|(uint64_t(1)<<24)|(uint64_t(1)<<25)))
            throw std::runtime_error("Dual FP32 output requires round-to-nearest and no flush-to-zero");
#endif
    }
    Transport& region_transport(){return transport_;}
    bool hardware() const override {return true;}
    void run(const int8_t*,const int8_t*,int32_t*,int,int,int) override {
        throw std::runtime_error("Dual region backend supports packet2 FP32 groups only; use the explicit ARM stem");
    }
    uint8_t* stream_fp32_arena(size_t bytes) override {
        if(!bytes || transport_.pending())throw std::runtime_error("Invalid or device-owned dual framework arena");
        if(base_) {
            if(bytes!=capacity_)throw std::runtime_error("Cannot resize bound dual framework tensors");
            return base_;
        }
        if(bytes>SIZE_MAX-63)throw std::runtime_error("Dual framework arena overflow");
        storage_.resize(bytes+63);capacity_=bytes;
        base_=reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(storage_.data())+63)&~uintptr_t(63));
        return base_;
    }
    void run_stream_fp32_packet(const std::vector<StreamOp>& ops,
            const std::vector<uint64_t>& packet,const int8_t* input,float* output) override {
        require_fp32_rounding();
        if(ops.empty() || ops.size()>6 || packet.empty() || !input || !output || transport_.pending())
            throw std::runtime_error("Invalid or unfinished dual ordinary submit");
        if(packet.size()*8<32+ops.size()*sizeof(StreamOp))
            throw std::runtime_error("Truncated dual packet operators");
        for(size_t i=0;i<ops.size();++i) {
            // stream_pack realigns weight/requant storage. Those offsets name
            // the validated packet, while graph shape/kind/activation stay fixed.
            if(std::memcmp(reinterpret_cast<const unsigned char*>(packet.data())+32+i*sizeof(StreamOp),
                           &ops[i],offsetof(StreamOp,weight_offset)))
                throw std::runtime_error("Dual packet operators differ from their graph descriptors");
        }
        const auto& first=ops.front();const auto& last=ops.back();
        if(first.ci<1 || first.ci>320 || first.h<1 || first.h>64 || first.w<1 || first.w>48 ||
            last.co<1 || last.co>320 || last.oh<1 || last.oh>64 || last.ow<1 || last.ow>48)
            throw std::runtime_error("Invalid dual ordinary shape");
        const uint64_t in_bytes=uint64_t(first.ci)*first.h*first.w;
        const uint64_t out_bytes=uint64_t(last.co)*last.oh*last.ow*4;
        const uintptr_t address=reinterpret_cast<uintptr_t>(output),base=reinterpret_cast<uintptr_t>(base_);
        if(!base_ || address<base || address-base>capacity_ || out_bytes>capacity_-(address-base) ||
            address%64 || packet.size()>U02_REGION_DUAL_PACKET_CAPACITY/8 ||
            in_bytes>U02_REGION_DUAL_INPUT_CAPACITY || out_bytes>U02_REGION_DUAL_OUTPUT_CAPACITY)
            throw std::runtime_error("Dual output is outside its bound framework arena");
        const auto begin=std::chrono::steady_clock::now();
        const auto job=transport_.run_legacy(packet.data(),packet.size()*8,input,in_bytes,
                                            ops.size(),output,out_bytes,30000000);
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        wait_ms+=job.elapsed_ns*1e-6;
#ifdef U02_CALL_DIAGNOSTIC
        diagnose("legacy_packet",ops,elapsed,job.elapsed_ns*1e-6,job.polls);
#endif
        // Includes private copies, ioctl and returned-output copy; do not call
        // this pure register-control or pure PL arithmetic time.
        control_ms+=std::max(0.0,elapsed-job.elapsed_ns*1e-6);poll_count+=job.polls;
        ++calls;++stream_calls;++stream_prepared_packet_calls;
        input_bytes+=packet.size()*8+in_bytes;output_bytes+=out_bytes;
        for(const auto& op:ops) {
            if(op.kind)++depthwise_calls;else ++convolution_calls;
            macs+=uint64_t(op.co)*op.oh*op.ow*(op.kind?9:op.ci);
        }
    }
};
