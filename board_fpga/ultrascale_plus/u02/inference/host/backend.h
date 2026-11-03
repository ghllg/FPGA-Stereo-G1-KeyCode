// Forked from board_single_pose e48cb5e for the light network.
#pragma once
#include <stdint.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <ostream>
#include "board_single_pose/hls/conv_contract.h"
#include "../hls/pair_contract.h"
#include "calibration.h"
#include "../hls/stream_contract.h"
#ifdef U02_STREAM_RANGE_CACHE
#include "stream_range_cache.h"
#endif

#ifdef U02_STREAM_REGION_HOOK
struct RegionScheduleExtension;
#endif
struct Backend {
#ifdef U02_STREAM_RANGE_CACHE
    StreamRangeCache range_cache;
#endif
    int cpu_threads=1;
    bool collect_calibration=false;
    std::vector<CalibrationRecord> calibration;
    void observe_input(const std::string& name,int channels,int height,int width,
                       const std::vector<float>& maxima,const std::vector<float>& scales) {
        if(!collect_calibration) return;
        if(maxima.size()!=size_t(channels) || (scales.size()!=1 && scales.size()!=size_t(channels)))
            throw std::runtime_error("Calibration shape mismatch");
        for(float v:maxima) if(!std::isfinite(v) || v<0) throw std::runtime_error("Invalid calibration maximum");
        for(float v:scales) if(!std::isfinite(v) || v<=0) throw std::runtime_error("Invalid calibration scale");
        calibration.push_back({name,channels,height,width,maxima,scales});
    }
    uint64_t calls=0, macs=0;
    uint64_t depthwise_calls=0;
    uint64_t fused_pair_calls=0;
    uint64_t stream_calls=0;
    uint64_t stream_prepared_packet_calls=0,stream_direct_input_calls=0;
    uint64_t stream_resident_packet_calls=0,stream_resident_prefix_copy_bytes=0;
    uint64_t stream_resident_model_packets=0,stream_resident_model_bytes=0,stream_resident_pool_bytes=0;
#ifdef U02_STREAM_TRACE
    unsigned stream_trace_group=0;
#endif
    bool fine_profile=false;
    double input_memcpy_ms=0,weight_memcpy_ms=0,output_memcpy_ms=0,metadata_memcpy_ms=0;
    double sync_cpu_ms=0,sync_device_ms=0,control_ms=0;
    double initial_sync_cpu_ms=0;
    uint64_t sync_cpu_bytes=0,sync_device_bytes=0,sync_cpu_calls=0,sync_device_calls=0,poll_count=0;
    double prepare_ms=0, wait_ms=0, readback_ms=0;
    uint64_t input_bytes=0, output_bytes=0;
    bool cached_dma=false;
    bool directional_dma=false;
    double custom_conv_ms=0, quantize_ms=0, im2col_ms=0, dequantize_ms=0;
#ifdef U02_STREAM_BOUNDARY_PROFILE
    struct BoundaryRow {
        int group,kind,ci,h,w,co,dynamic_coefficients,zero_channels,zero_elements;
        double range_ms,quant_ms,coeff_ms,patch_ms;
    };
    std::vector<BoundaryRow> boundary_rows;
    double boundary_range_ms=0,boundary_quant_ms=0,boundary_coeff_ms=0,boundary_patch_ms=0;
    void write_boundary_profile(std::ostream& out) const {
        out<<",\"boundary_profile_scope\":\"last_complete_forward_wall_time_including_scheduling_not_cpu_cycles\",\"boundary_groups\":[";
        for(size_t i=0;i<boundary_rows.size();++i) {
            const auto& r=boundary_rows[i];if(i)out<<",";
            out<<"{\"group\":"<<r.group<<",\"kind\":\""<<(r.kind?"DW":"PW")
               <<"\",\"input_chw\":["<<r.ci<<","<<r.h<<","<<r.w<<"],\"first_co\":"<<r.co
               <<",\"dynamic_coefficients\":"<<r.dynamic_coefficients
               <<",\"zero_channels\":"<<r.zero_channels<<",\"zero_elements\":"<<r.zero_elements
               <<",\"range_ms\":"<<r.range_ms<<",\"quant_ms\":"<<r.quant_ms
               <<",\"coeff_ms\":"<<r.coeff_ms<<",\"patch_ms\":"<<r.patch_ms<<"}";
        }
        out<<"]";
    }
#endif
    uint64_t conv_layers=0;
    uint64_t cpu_selected_layers=0;
    double cpu_selected_ms=0;
    uint64_t direct_calls=0;
    uint64_t convolution_calls=0,resident_reuses=0;
    uint64_t pl_quantize_calls=0;
#ifdef U02_CALL_DIAGNOSTIC
    struct DiagnosticCall {
        int node; std::string name, kind; std::vector<StreamOp> ops;
        double host_ms,driver_wait_ms;uint64_t polls;
    };
    int diagnostic_node=-1;std::string diagnostic_name;
    std::vector<DiagnosticCall> diagnostic_calls;
    void diagnose(const char* kind,const std::vector<StreamOp>& ops,double host_ms,double wait_ms,uint64_t polls) {
        diagnostic_calls.push_back({diagnostic_node,diagnostic_name,kind,ops,host_ms,wait_ms,polls});
    }
    void write_call_diagnostics(std::ostream& out) const {
        out<<",\"call_diagnostic_scope\":\"last_complete_forward_host_and_driver_timing_not_hardware_cycles\",\"pl_call_diagnostics\":[";
        for(size_t i=0;i<diagnostic_calls.size();++i) {
            const auto& row=diagnostic_calls[i];if(i)out<<",";
            out<<"{\"sequence\":"<<i<<",\"node\":"<<row.node<<",\"name\":\""<<row.name
               <<"\",\"kind\":\""<<row.kind<<"\",\"host_ms\":"<<row.host_ms
               <<",\"driver_wait_ms\":"<<row.driver_wait_ms<<",\"polls\":"<<row.polls
               <<",\"hardware_start_done_cycles\":null,\"axi_transactions\":null,\"axi_bursts\":null"
               <<",\"cache_sync_ms\":null,\"traffic_scope\":\"logical_operator_bytes_not_observed_axi_traffic\",\"operators\":[";
            for(size_t j=0;j<row.ops.size();++j) {
                const auto& p=row.ops[j];if(j)out<<",";
                out<<"{\"index_within_call\":"<<j<<",\"type\":\""<<(p.kind?"DW":"PW")
                   <<"\",\"input_chw\":["<<p.ci<<","<<p.h<<","<<p.w<<"],\"output_chw\":["
                   <<p.co<<","<<p.oh<<","<<p.ow<<"],\"macs\":"<<uint64_t(p.co)*p.oh*p.ow*(p.kind?9:p.ci)
                   <<",\"int8_weight_bytes\":"<<uint64_t(p.co)*(p.kind?9:p.ci)
                   <<",\"int8_input_bytes\":"<<uint64_t(p.ci)*p.h*p.w
                   <<",\"int8_output_bytes\":"<<uint64_t(p.co)*p.oh*p.ow<<"}";
            }
            out<<"]}";
        }
        out<<"]";
    }
#endif
    void reset_forward_metrics() {
#ifdef U02_STREAM_BOUNDARY_PROFILE
        boundary_rows.clear();boundary_rows.reserve(110);
        boundary_range_ms=boundary_quant_ms=boundary_coeff_ms=boundary_patch_ms=0;
#endif
#ifdef U02_STREAM_RANGE_CACHE
        range_cache.begin_forward();
#endif
#ifdef U02_CALL_DIAGNOSTIC
        diagnostic_calls.clear();
#endif
        calibration.clear();
        stream_calls=0;
        stream_prepared_packet_calls=stream_direct_input_calls=0;
        stream_resident_packet_calls=stream_resident_prefix_copy_bytes=0;
        calls=macs=depthwise_calls=fused_pair_calls=0;
        sync_cpu_bytes=sync_device_bytes=sync_cpu_calls=sync_device_calls=poll_count=0;
        input_bytes=output_bytes=conv_layers=cpu_selected_layers=direct_calls=0;
        convolution_calls=resident_reuses=pl_quantize_calls=0;
        input_memcpy_ms=weight_memcpy_ms=output_memcpy_ms=metadata_memcpy_ms=0;
        sync_cpu_ms=sync_device_ms=control_ms=prepare_ms=wait_ms=readback_ms=0;
        custom_conv_ms=quantize_ms=im2col_ms=dequantize_ms=cpu_selected_ms=0;
#ifdef U02_STREAM_FP32_OUTPUT
        async_submissions=async_span_waits=async_heap_inputs=0;
#endif
        // Configuration, initial setup cost and derived DMA/resident state persist.
    }
    virtual ~Backend() {}
    virtual void run(const int8_t* a, const int8_t* b, int32_t* c, int m, int n, int k)=0;
    // Optional borrowed DMA storage. Input is writable until run_prepared;
    // returned results remain valid only until the next backend operation.
    // nullptr means the caller must use run with its own buffers.
    virtual int8_t* prepare_input(int m,int n,int k) {return nullptr;}
    virtual const int32_t* run_prepared(const int8_t* a) {
        throw std::runtime_error("Prepared input unsupported by this backend");
    }
    // first=true starts a new immutable weight/image pair. Later chunks may
    // change only start and N; all other fields and input storage stay valid.
    virtual const int32_t* run_convolution(const int8_t* a,const int8_t* image,
        int32_t* scratch,int m,int n,int k,const PoseConv& shape,bool first) {
        throw std::runtime_error("Direct convolution unsupported by this backend");
    }
    virtual const int32_t* run_depthwise(const int8_t*,const int8_t*,int32_t*,int,int,int,const PoseConv&) {
        throw std::runtime_error("Depthwise not supported");
    }
    virtual bool hardware() const=0;
    // Writable CPU-owned input until the next synchronous streaming submit.
    // The kernel still performs its normal DMA ownership transfer at RUN.
    virtual int8_t* prepare_stream_input(size_t) {return nullptr;}
    virtual void register_stream_packet(int,const std::vector<uint64_t>&) {}
#ifdef U02_STREAM_REGION_HOOK
    virtual void register_stream_boundary(int,const std::vector<float>&,const std::vector<float>&,
                                         const std::vector<float>&) {}
    virtual RegionScheduleExtension* region_extension(){return nullptr;}
#endif
#ifdef U02_STREAM_FP32_OUTPUT
    virtual void synchronize_span(const void*,size_t) {}
    virtual void finish_forward() {}
    uint64_t async_submissions=0,async_span_waits=0,async_heap_inputs=0;
    virtual uint8_t* stream_fp32_arena(size_t) {return nullptr;}
    virtual void run_stream_fp32_packet(const std::vector<StreamOp>&,
                            const std::vector<uint64_t>&,const int8_t*,float*) {
        throw std::runtime_error("FP32 DMA output unsupported");
    }
#endif
    virtual const int32_t* run_stream_packet(const std::vector<StreamOp>&,
                            const std::vector<uint64_t>&,const int8_t*) {
        throw std::runtime_error("Prepared streaming packet unsupported");
    }
    // Returned output remains valid until the next backend submission. CPU
    // reference backends fill the supplied scratch; coherent DMA may lend its
    // completed result buffer. Callers must finish reading before resubmitting.
    virtual const int32_t* run_stream(const std::vector<StreamOp>&,const std::vector<int8_t>&,
                            const std::vector<StreamRequant>&,const int8_t*,int32_t*) {
        throw std::runtime_error("Streaming graph backend is not deployed on this hardware");
    }
    virtual const int32_t* run_pw_dw(const int8_t*,const int8_t*,const int8_t*,
        const LightAffine*,int32_t*,int,const PoseConv&) {
        throw std::runtime_error("Fused PW/DW not supported by this backend");
    }
    virtual const int32_t* run_quantized_convolution(const int8_t* a,const float* image,
        size_t channel_stride,const uint32_t* thresholds,int32_t* scratch,
        int m,int n,int k,const PoseConv& shape,bool first) {
        throw std::runtime_error("PL quantization unsupported by this backend");
    }
};
std::unique_ptr<Backend> make_backend(const std::string& mode,
    const std::string& uio="/dev/uio0", const std::string& dma="udmabuf0");
