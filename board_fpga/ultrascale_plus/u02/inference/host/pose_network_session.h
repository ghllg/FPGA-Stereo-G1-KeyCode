#pragma once
#include "pl_conv_adapter.h"
#include "pl_pair.h"
#include "pl_stream.h"
#include "stream_schedule.h"
#include "backend_stream_dual.h"
#ifdef U02_STREAM_PIPELINE
#include "backend_stream_pipeline.h"
#endif
#ifdef U02_STREAM_DUAL_FP32
#include "backend_stream_dual_fp32.h"
#endif
#include <fstream>
#include <iomanip>

// Reusable whole-network state, one session per worker. No shared NCNN Net,
// schedule, mutable quantization scratch, packet pool, or output storage.
struct PoseNetworkSession {
    std::unique_ptr<Backend> backend,stem;
    ncnn::Net net;
    std::unique_ptr<StreamSchedule> schedule;
    ncnn::Mat output;
    struct Timing {
        uint64_t start_ns=0,end_ns=0,calls=0,macs=0;
        double prepare_ms=0,wait_ms=0,queue_ms=0,control_ms=0,quantize_ms=0,dequantize_ms=0;
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
        uint64_t forward_thread_cpu_ns=0;
        U02IoctlCpuProfile ioctl_cpu;
#endif
#ifdef U02_STREAM_BOUNDARY_PROFILE
        double boundary_range_ms=0,boundary_quant_ms=0,boundary_coeff_ms=0,boundary_patch_ms=0;
#endif
#ifdef U02_STREAM_RANGE_CACHE
        uint64_t range_hit_elements=0,range_verified_elements=0,range_published_elements=0;
#endif
    } timing;
    static uint64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    PoseNetworkSession(const std::string& mode,const char* param,const char* weights,int threads,
            std::unique_ptr<Backend> custom=std::unique_ptr<Backend>()) :backend(std::move(custom)) {
        if(mode!="cpu" && mode!="pl-stream" && mode!="stream-csim")
            throw std::runtime_error("Session needs full cpu/pl-stream/stream-csim backend");
        if(threads<1 || threads>4)throw std::runtime_error("Invalid session thread count");
        net.opt.num_threads=threads;net.opt.use_vulkan_compute=false;
        net.opt.use_packing_layout=(mode=="cpu");net.opt.use_fp16_storage=false;
        net.opt.use_fp16_arithmetic=false;net.opt.use_bf16_storage=false;
        if(mode!="cpu") {
            if(!backend) {
                if(mode=="pl-stream")throw std::runtime_error("PL session requires an explicit owned context backend");
                backend=make_backend(mode);
            }
            stem=make_backend(mode=="pl-stream"?"arm-stem":"csim");stem->cpu_threads=threads;
            net.register_custom_layer("Convolution",create_conv,nullptr,stem.get());
            net.register_custom_layer("LightPwDw",create_pair,nullptr,backend.get());
            net.register_custom_layer("U02Stream",create_stream,nullptr,backend.get());
        } else if(backend)throw std::runtime_error("CPU session must not have a PL backend");
        if(net.load_param(param) || net.load_model(weights))throw std::runtime_error("Session model load failed");
        if(mode!="cpu")schedule.reset(new StreamSchedule(net,param
#ifdef U02_STREAM_FP32_OUTPUT
            ,backend.get()
#endif
        ));
#ifdef U02_STREAM_RANGE_CACHE
        if(backend)backend->range_cache.bind_param(STREAM_SCHEDULE_PARAM);
#endif
    }
    void forward(const ncnn::Mat& input) {
        if(backend)backend->reset_forward_metrics();
        if(stem)stem->reset_forward_metrics();
        auto* dual=dynamic_cast<StreamDualBackend*>(backend.get());if(dual)dual->queue_wait_ms=0;
#ifdef U02_STREAM_PIPELINE
        auto* pipeline=dynamic_cast<StreamPipelineBackend*>(backend.get());if(pipeline)pipeline->queue_wait_ms=0;
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
        if(pipeline)pipeline->ioctl_cpu.reset();
#endif
#endif
        timing=Timing();timing.start_ns=now_ns();
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
        const auto forward_cpu_start=u02_thread_cpu_ns();
#endif
        ncnn::Mat result;
        if(schedule)schedule->forward(input,result);
        else {
            auto ex=net.create_extractor();
            if(ex.input("image",input) || ex.extract("conv2d_441.tmp_1",result))
                throw std::runtime_error("Session full network inference failed");
        }
        timing.end_ns=now_ns();
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
        const auto forward_cpu_end=u02_thread_cpu_ns();
        if(forward_cpu_end<forward_cpu_start)throw std::runtime_error("Forward CPU clock reversed");
        timing.forward_thread_cpu_ns=forward_cpu_end-forward_cpu_start;
        if(pipeline)timing.ioctl_cpu=pipeline->ioctl_cpu;
#endif
        if(result.dims!=3 || result.w!=24 || result.h!=32 || result.c!=17 || result.elempack!=1)
            throw std::runtime_error("Session expected complete COCO17 heatmap");
        // CPU extraction and schedule storage have different lifetimes; retain
        // a complete private copy outside the forward timing in both modes.
        output=result.clone();if(output.empty())throw std::runtime_error("Session output copy failed");
        for(int c=0;c<17;++c)for(int i=0;i<768;++i)
            if(!std::isfinite(static_cast<const float*>(output.channel(c))[i]))
                throw std::runtime_error("Nonfinite session heatmap");
        if(backend) {
            timing.calls=backend->calls;timing.macs=backend->macs;timing.wait_ms=backend->wait_ms;timing.prepare_ms=backend->prepare_ms;
            timing.control_ms=backend->control_ms;timing.quantize_ms=backend->quantize_ms;
            timing.dequantize_ms=backend->dequantize_ms;
#ifdef U02_STREAM_BOUNDARY_PROFILE
            timing.boundary_range_ms=backend->boundary_range_ms;timing.boundary_quant_ms=backend->boundary_quant_ms;
            timing.boundary_coeff_ms=backend->boundary_coeff_ms;timing.boundary_patch_ms=backend->boundary_patch_ms;
#endif
#ifdef U02_STREAM_RANGE_CACHE
            timing.range_hit_elements=backend->range_cache.hit_elements;
            timing.range_verified_elements=backend->range_cache.verified_elements;
            timing.range_published_elements=backend->range_cache.published_elements;
#endif
        }
        if(dual)timing.queue_ms=dual->queue_wait_ms;
#ifdef U02_STREAM_PIPELINE
        if(pipeline)timing.queue_ms=pipeline->queue_wait_ms;
#endif
    }
    void save(const std::string& path,int source_w,int source_h,double rx,double ry,double rw,double rh) const {
        std::ofstream json(path),heat(path+".heatmap",std::ios::binary);
        if(!json || !heat)throw std::runtime_error("Cannot save complete session outputs");
        json<<std::setprecision(9)<<"{\"scope\":\"last_complete_forward\",\"coordinate_space\":\"source_image_pixels\","
            <<"\"keypoint_order\":\"COCO17\",\"source_size\":["<<source_w<<","<<source_h
            <<"],\"roi\":["<<rx<<","<<ry<<","<<rw<<","<<rh<<"],\"pl_calls\":"<<timing.calls
            <<",\"keypoints\":[";
        for(int c=0;c<17;++c) {
            const float* h=output.channel(c);int best=0;
            for(int i=0;i<768;++i)if(h[i]>h[best])best=i;
            float x=best%24,y=best/24;
            if(x>0 && x<23 && y>0 && y<31) {
                const float dx=h[best+1]-h[best-1],dy=h[best+24]-h[best-24];
                x+=dx>0?.25f:dx<0?-.25f:0;y+=dy>0?.25f:dy<0?-.25f:0;
            }
            if(c)json<<",";
            json<<"["<<float(rx+x*rw/24.0)<<","<<float(ry+y*rh/32.0)<<","<<h[best]<<"]";
            heat.write(reinterpret_cast<const char*>(h),768*4);
        }
        json<<"]";
#ifdef U02_STREAM_BOUNDARY_PROFILE
        if(backend)backend->write_boundary_profile(json);
#endif
        json<<"}\n";
        if(!json || !heat)throw std::runtime_error("Session output write failed");
    }
    std::vector<float> points(double rx,double ry,double rw,double rh) const {
        if(output.dims!=3 || output.w!=24 || output.h!=32 || output.c!=17 || output.elempack!=1)
            throw std::runtime_error("Expected complete output before keypoint decoding");
        std::vector<float> xy(51);
        for(int c=0;c<17;++c) {
            const float* h=output.channel(c);int best=0;
            for(int i=0;i<768;++i)if(h[i]>h[best])best=i;
            float x=best%24,y=best/24;
            if(x>0 && x<23 && y>0 && y<31) {
                const float dx=h[best+1]-h[best-1],dy=h[best+24]-h[best-24];
                x+=dx>0?.25f:dx<0?-.25f:0;y+=dy>0?.25f:dy<0?-.25f:0;
            }
            xy[c*3]=float(rx+x*rw/24.0);xy[c*3+1]=float(ry+y*rh/32.0);xy[c*3+2]=h[best];
        }
        return xy;
    }
};
