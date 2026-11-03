#include "pl_stream.h"
#include "backend.h"
#include "quantize.h"
#include "stream_quantize.h"
#include "stream_quantize_bounded.h"
#include "stream_dequantize.h"
#ifdef U02_STREAM_RANGE_CACHE
#include "stream_dequantize_range.h"
#endif
#include "stream_affine_coeff.h"
#include "stream_wire.h"
#ifdef U02_STREAM_FP32_OUTPUT
#include "stream_fp32_patch.h"
#endif
#ifdef U02_STREAM_EXPORT_FP32_FIXTURE
#include "stream_fp32_wire.h"
#include <fstream>
#endif
#include <net.h>
#include <modelbin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfenv>
#include <iostream>
#include "stream_model_generated.h"
#if defined(U02_STREAM_BOUNDARY_PROFILE) && (defined(U02_STREAM_DW_CHANNEL_LOCAL) || defined(U02_STREAM_MERGED_BOUNDARY_WORK) || !defined(U02_STREAM_PARALLEL_AFFINE))
#error Boundary profiling requires separate range, quantization and coefficient phases
#endif

// The same NCNN layer supplies the integer reference and the future PL backend.
// Current V06f hardware explicitly rejects run_stream; there is no CPU fallback.
class PlStream : public ncnn::Layer {
#ifdef U02_STREAM_PARALLEL_AFFINE
    static uint64_t coefficient_fp_environment() {
#ifdef __aarch64__
        uint64_t value;asm volatile("mrs %0, fpcr" : "=r"(value));return value;
#else
        return uint64_t(std::fegetround());
#endif
    }
#endif
    Backend* backend;
    int group_id=-1;
    std::vector<StreamOp> ops;
    std::vector<int8_t> weights;
    std::vector<std::vector<float>> weight_scales,bias,input_scales;
    std::vector<StreamRequant> fixed_coeff;
    // A network/backend session submits synchronously. Its boundary scratch
    // remains private to this layer and is reused across complete forwards.
    mutable std::vector<float> boundary_buffer,maximum_buffer;
    mutable std::vector<int8_t> quantized_buffer;
    mutable std::vector<uint64_t> prepared_packet;
#ifdef U02_STREAM_RANGE_CACHE
    mutable std::vector<unsigned char> range_hits,range_published;
    bool input_maximum(const float* src,int count,int channel,float& maximum) const {
        if(backend->range_cache.get(group_id,channel,stream_range_fpcr(),maximum)) {
            if(backend->range_cache.verify) {
                float checked=0;
                if(!pose_max_abs(src,count,checked) || std::memcmp(&checked,&maximum,4))return false;
            }
            range_hits[channel]=1;return true;
        }
        return pose_max_abs(src,count,maximum);
    }
#else
    bool input_maximum(const float* src,int count,int,float& maximum) const {
        return pose_max_abs(src,count,maximum);
    }
#endif
#ifdef U02_STREAM_FP32_OUTPUT
    mutable std::vector<StreamOutputAffine> output_affine;
#endif
#ifdef U02_STREAM_PARALLEL_AFFINE
    mutable std::vector<StreamRequant> coeff_buffer;
#endif
    void make_coeff(size_t i,const std::vector<float>& incoming,
                    std::vector<StreamRequant>& coeff) const {
        const auto& op=ops[i];const auto& outgoing=input_scales[i+1];
        if(coeff.size()<size_t(op.requant_offset+op.co)) coeff.resize(op.requant_offset+op.co);
        for(int ch=0;ch<op.co;++ch) {
            const double scale_in=incoming[op.kind?ch:0],scale_out=outgoing.size()==1?outgoing[0]:outgoing[ch];
            const double ratio=scale_in*weight_scales[i][ch]/scale_out;
            if(!std::isfinite(ratio) || ratio<0) throw std::runtime_error("Invalid stream affine ratio");
            // Start at the exponent-derived bound, then retain the exact old
            // ldexp comparison at the 24-bit boundary. Avoid up to 40 calls
            // to ldexp per channel on every forward.
            int shift=stream_affine_shift(ratio);
            while(shift>0 && stream_affine_scale(ratio,shift)>8388607.0)--shift;
            const double multiplier=stream_affine_scale(ratio,shift),offset=stream_affine_scale(double(bias[i][ch])/scale_out,shift);
            if(!std::isfinite(multiplier+offset) || multiplier<0 || multiplier>8388607 || std::abs(offset)>double(int64_t(1)<<60))
                throw std::runtime_error("Stream affine coefficient exceeds contract");
            coeff[op.requant_offset+ch]={int32_t(stream_affine_round(multiplier)),shift,stream_affine_round(offset)};
        }
    }
#ifdef U02_STREAM_PARALLEL_AFFINE
    // Each worker writes one already allocated channel; identical arithmetic
    // to make_coeff, without resizing or throwing across an OpenMP boundary.
    void make_boundary_coeff(int ch,const std::vector<float>& incoming,
                             std::vector<StreamRequant>& coeff) const {
        const auto& op=ops.front();const auto& outgoing=input_scales[1];
        const double scale_in=incoming[op.kind?ch:0],scale_out=outgoing.size()==1?outgoing[0]:outgoing[ch];
        const double ratio=scale_in*weight_scales[0][ch]/scale_out;
        if(!std::isfinite(ratio) || ratio<0)throw std::runtime_error("Invalid stream affine ratio");
        int shift=stream_affine_shift(ratio);
        while(shift>0 && stream_affine_scale(ratio,shift)>8388607.0)--shift;
        const double multiplier=stream_affine_scale(ratio,shift),offset=stream_affine_scale(double(bias[0][ch])/scale_out,shift);
        if(!std::isfinite(multiplier+offset) || multiplier<0 || multiplier>8388607 || std::abs(offset)>double(int64_t(1)<<60))
            throw std::runtime_error("Stream affine coefficient exceeds contract");
        coeff[op.requant_offset+ch]={int32_t(stream_affine_round(multiplier)),shift,stream_affine_round(offset)};
    }
#endif
public:
    explicit PlStream(Backend* b):backend(b) {one_blob_only=true;}
    int load_param(const ncnn::ParamDict& p) override {
        group_id=p.get(0,-1);
        if(group_id<0 || group_id>=STREAM_MODEL_GROUP_COUNT) return -1;
        const auto& group=STREAM_MODEL_GROUPS[group_id];
        for(int i=0;i<group.count;++i) {
            const auto& spec=STREAM_MODEL_OPS[group.first+i];
            if(!stream_op_valid(spec.op)) return -1;
            ops.push_back(spec.op);
            input_scales.emplace_back(STREAM_MODEL_SCALES+spec.scale_offset,STREAM_MODEL_SCALES+spec.scale_offset+spec.scale_count);
        }
        return 0;
    }
    int load_model(const ncnn::ModelBin& mb) override {
        const auto& group=STREAM_MODEL_GROUPS[group_id];
        for(size_t i=0;i<ops.size();++i) {
            const auto& op=ops[i];const int k=op.kind?9:op.ci,count=op.co*k;
            ncnn::Mat raw=mb.load(count,0);if(raw.empty() || raw.elemsize!=4) return -1;
            std::vector<float> offsets(op.co,0),scales(op.co);
            if(STREAM_MODEL_OPS[group.first+i].bias_term) {
                ncnn::Mat b=mb.load(op.co,1);if(b.empty()) return -1;
                std::copy((const float*)b,(const float*)b+op.co,offsets.begin());
            }
            const float* src=raw;weights.resize(op.weight_offset+count);
            for(int ch=0;ch<op.co;++ch) {
                float maximum=0;
                for(int q=0;q<k;++q) {if(!std::isfinite(src[ch*k+q]))return -1;maximum=std::max(maximum,std::abs(src[ch*k+q]));}
                if(!std::isfinite(offsets[ch])) return -1;
                scales[ch]=maximum?maximum/127.f:1.f;
                for(int q=0;q<k;++q) weights[op.weight_offset+ch*k+q]=pose_quantize(src[ch*k+q],scales[ch]);
            }
            weight_scales.push_back(scales);bias.push_back(offsets);
        }
        // Only the first operator uses the runtime boundary scale. All
        // internal scales, weights and biases are immutable model data.
        for(size_t i=1;i+1<ops.size();++i) make_coeff(i,input_scales[i],fixed_coeff);
        const auto& first=ops.front();
        boundary_buffer.resize(first.kind?first.ci:1);
        maximum_buffer.resize(first.ci);
#ifdef U02_STREAM_RANGE_CACHE
        const auto& range_last=ops.back();
        backend->range_cache.register_group(group_id,first.ci,first.h,first.w,range_last.co,range_last.oh,range_last.ow);
        range_hits.resize(first.ci);range_published.resize(range_last.co);
#endif
        quantized_buffer.resize(size_t(first.ci)*first.h*first.w);
#ifdef U02_STREAM_PARALLEL_AFFINE
        coeff_buffer=fixed_coeff;
        if(ops.size()>1)make_coeff(0,input_scales.front(),coeff_buffer);
#endif
        if(backend->hardware()) {
            std::vector<StreamRequant> initial=fixed_coeff;
            if(ops.size()>1)make_coeff(0,input_scales.front(),initial);
#ifdef U02_STREAM_FP32_OUTPUT
            const auto& last=ops.back();
            const auto& incoming=ops.size()==1?input_scales.front():input_scales.back();
            output_affine.resize(last.co);
            for(int ch=0;ch<last.co;++ch)
                output_affine[ch]={incoming[last.kind?ch:0],weight_scales.back()[ch],bias.back()[ch],0};
            prepared_packet=stream_pack_fp32_output(ops,weights,initial,output_affine);
#else
            prepared_packet=stream_pack(ops,weights,initial);
#endif
#if defined(U02_STREAM_RESIDENT_PACKETS) || defined(U02_STREAM_REGION_HOOK)
            backend->register_stream_packet(group_id,prepared_packet);
#endif
#ifdef U02_STREAM_REGION_HOOK
            const std::vector<float> outgoing=ops.size()>1?input_scales[1]:std::vector<float>(1,1.f);
            backend->register_stream_boundary(group_id,weight_scales.front(),bias.front(),outgoing);
#endif
        }
        return 0;
    }
    int forward(const ncnn::Mat& in,ncnn::Mat& out,const ncnn::Option& opt) const override {
        try {
            const auto& first=ops.front(); const auto& last=ops.back();
            if(in.dims!=3 || in.c!=first.ci || in.h!=first.h || in.w!=first.w || in.elempack!=1 || in.elemsize!=4) return -1;
            if(backend->collect_calibration) throw std::runtime_error("Use unfused graph for calibration observations");
            auto qbegin=std::chrono::steady_clock::now();
#ifdef U02_STREAM_PROFILE_PHASES
            const double wait_before=backend->wait_ms,control_before=backend->control_ms;
            const uint64_t polls_before=backend->poll_count;
#endif
            auto& boundary=boundary_buffer;
            auto& channel_maximum=maximum_buffer;
            auto& image=quantized_buffer;
            int8_t* quantized=backend->prepare_stream_input(image.size());
            if(!quantized)quantized=image.data();
            std::fill(boundary.begin(),boundary.end(),0.f);
            std::fill(channel_maximum.begin(),channel_maximum.end(),0.f);
#ifdef U02_STREAM_RANGE_CACHE
            std::fill(range_hits.begin(),range_hits.end(),0);
            std::fill(range_published.begin(),range_published.end(),0);
#endif
            int invalid_input=0;
#ifdef U02_STREAM_PARALLEL_AFFINE
            auto& coeff=coeff_buffer;
            int invalid_coeff=0;
            int coefficient_env_mismatch=0;
            const uint64_t coefficient_env=coefficient_fp_environment();
#endif
            int boundary_threads=opt.num_threads;
#ifdef U02_STREAM_SMALL_TENSOR_SERIAL
            // Small boundaries do not amortize OpenMP team/barrier costs.
            // Preserve channel order, exact division and finite-input checks.
            if(image.size()<=4096) boundary_threads=1;
#endif
#if defined(U02_STREAM_PROFILE_PHASES) || defined(U02_STREAM_BOUNDARY_PROFILE)
            std::chrono::steady_clock::time_point range_end;
#endif
#ifdef U02_STREAM_BOUNDARY_PROFILE
            std::chrono::steady_clock::time_point quant_only_end;
#endif
            // Reuse one team for both passes. The implicit for/single barriers
            // publish all maxima and scales before any thread quantizes.
#ifdef U02_STREAM_DW_CHANNEL_LOCAL
            // Depthwise scales and first-operation affine coefficients are
            // channel-local. Publish each channel's maximum, quantized plane
            // and coefficient in one worker, with only the final team join.
            // PW retains its required whole-tensor maximum and barriers.
            if(first.kind==1 && first.co==first.ci) {
#ifdef U02_STREAM_PARALLEL_AFFINE
                #pragma omp parallel for num_threads(boundary_threads) reduction(|:invalid_input,invalid_coeff,coefficient_env_mismatch)
#else
                #pragma omp parallel for num_threads(boundary_threads) reduction(|:invalid_input)
#endif
                for(int ch=0;ch<first.ci;++ch) {
                    float maximum=0;
                    if(!input_maximum(in.channel(ch),in.w*in.h,ch,maximum)){invalid_input=1;continue;}
                    channel_maximum[ch]=maximum;
                    boundary[ch]=maximum?maximum/127.f:1.f;
                    stream_quantize_bounded_array(in.channel(ch),quantized+ch*first.h*first.w,
                        first.h*first.w,boundary[ch],channel_maximum[ch]);
#ifdef U02_STREAM_PARALLEL_AFFINE
                    if(coefficient_fp_environment()!=coefficient_env)coefficient_env_mismatch=1;
                    else if(ops.size()>1) {
                        try{make_boundary_coeff(ch,boundary,coeff);}
                        catch(const std::exception&){invalid_coeff=1;}
                    }
#endif
                }
#ifdef U02_STREAM_PROFILE_PHASES
                range_end=std::chrono::steady_clock::now();
#endif
            } else
#endif
            {
#ifdef U02_STREAM_PARALLEL_AFFINE
            #pragma omp parallel num_threads(boundary_threads) reduction(|:coefficient_env_mismatch)
#else
            #pragma omp parallel num_threads(boundary_threads)
#endif
            {
#ifdef U02_STREAM_PARALLEL_AFFINE
                const bool coefficient_env_matches=coefficient_fp_environment()==coefficient_env;
                if(!coefficient_env_matches)coefficient_env_mismatch=1;
#endif
                #pragma omp for reduction(|:invalid_input)
                for(int ch=0;ch<first.ci;++ch) {
                    if(!input_maximum(in.channel(ch),in.w*in.h,ch,channel_maximum[ch])) invalid_input=1;
                }
                #pragma omp single
                {
                    if(!invalid_input) {
                        // Preserve the original channel order and exact scale.
                        for(int ch=0;ch<first.ci;++ch) {
                            float& slot=boundary[first.kind?ch:0];slot=std::max(slot,channel_maximum[ch]);
                        }
                        for(float& s:boundary) s=s?s/127.f:1.f;
                    }
#if defined(U02_STREAM_PROFILE_PHASES) || defined(U02_STREAM_BOUNDARY_PROFILE)
                    range_end=std::chrono::steady_clock::now();
#endif
                }
#if defined(U02_STREAM_MERGED_BOUNDARY_WORK) && defined(U02_STREAM_PARALLEL_AFFINE)
                // Scales and invalid_input are published by the preceding
                // single barrier. Input quantization and first-op coefficient
                // preparation have disjoint destinations, so share one work
                // distribution and final barrier instead of two adjacent ones.
                const int coefficient_channels=ops.size()>1?first.co:0;
                const int work_channels=std::max(first.ci,coefficient_channels);
                #pragma omp for reduction(|:invalid_coeff)
                for(int ch=0;ch<work_channels;++ch) {
                    if(!invalid_input && ch<first.ci)
                        stream_quantize_bounded_array(in.channel(ch),quantized+ch*first.h*first.w,first.h*first.w,boundary[first.kind?ch:0],channel_maximum[ch]);
                    if(!invalid_input && ch<coefficient_channels && coefficient_env_matches) {
                        try {make_boundary_coeff(ch,boundary,coeff);}
                        catch(const std::exception&) {invalid_coeff=1;}
                    }
                }
#else
                #pragma omp for
                for(int ch=0;ch<first.ci;++ch) {
                    if(!invalid_input)
                        stream_quantize_bounded_array(in.channel(ch),quantized+ch*first.h*first.w,first.h*first.w,boundary[first.kind?ch:0],channel_maximum[ch]);
                }
#ifdef U02_STREAM_BOUNDARY_PROFILE
                #pragma omp single
                {quant_only_end=std::chrono::steady_clock::now();}
#endif
#ifdef U02_STREAM_PARALLEL_AFFINE
                #pragma omp for reduction(|:invalid_coeff)
                for(int ch=0;ch<(ops.size()>1?first.co:0);++ch) {
                    if(!invalid_input && coefficient_env_matches) {
                        try {make_boundary_coeff(ch,boundary,coeff);}
                        catch(const std::exception&) {invalid_coeff=1;}
                    }
                }
#endif
#endif
            }
            }
            if(invalid_input) return -1;
#ifdef U02_STREAM_RANGE_CACHE
            for(unsigned char hit:range_hits)if(hit) {
                backend->range_cache.hit_elements+=first.h*first.w;
                if(backend->range_cache.verify)backend->range_cache.verified_elements+=first.h*first.w;
            }
#endif
#ifdef U02_STREAM_PARALLEL_AFFINE
            // Existing OpenMP workers may retain an earlier FPCR. Preserve
            // serial coefficient semantics if their environment differs.
            if(coefficient_env_mismatch && ops.size()>1)make_coeff(0,boundary,coeff);
            else if(invalid_coeff)throw std::runtime_error("Parallel stream affine coefficient exceeds contract");
#endif
#if defined(U02_STREAM_PROFILE_PHASES) || defined(U02_STREAM_BOUNDARY_PROFILE)
            const auto quant_end=std::chrono::steady_clock::now();
#endif
#ifndef U02_STREAM_PARALLEL_AFFINE
            std::vector<StreamRequant> coeff=fixed_coeff;
            if(ops.size()>1) make_coeff(0,boundary,coeff);
#endif
            if(backend->hardware()) {
#ifdef U02_STREAM_FP32_OUTPUT
                if(ops.size()==1)
                    for(int ch=0;ch<last.co;++ch)output_affine[ch].incoming=boundary[last.kind?ch:0];
                stream_patch_fp32(prepared_packet,ops,coeff,output_affine);
#else
                stream_patch_requant(prepared_packet,coeff);
#endif
            }
#if defined(U02_STREAM_PROFILE_PHASES) || defined(U02_STREAM_BOUNDARY_PROFILE)
            const auto coeff_end=std::chrono::steady_clock::now();
#endif
#ifdef U02_STREAM_BOUNDARY_PROFILE
            const double range_ms=std::chrono::duration<double,std::milli>(range_end-qbegin).count();
            const double quant_ms=std::chrono::duration<double,std::milli>(quant_only_end-range_end).count();
            const double coeff_ms=std::chrono::duration<double,std::milli>(quant_end-quant_only_end).count();
            const double patch_ms=std::chrono::duration<double,std::milli>(coeff_end-quant_end).count();
            int zero_channels=0;
            for(float maximum:channel_maximum) {
                uint32_t bits;std::memcpy(&bits,&maximum,sizeof(bits));
                if(bits==0)++zero_channels;
            }
            backend->boundary_rows.push_back({group_id,first.kind,first.ci,first.h,first.w,first.co,
                ops.size()>1?first.co:0,zero_channels,zero_channels*first.h*first.w,range_ms,quant_ms,coeff_ms,patch_ms});
            backend->boundary_range_ms+=range_ms;backend->boundary_quant_ms+=quant_ms;
            backend->boundary_coeff_ms+=coeff_ms;backend->boundary_patch_ms+=patch_ms;
#endif
            backend->quantize_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-qbegin).count();
            std::vector<int32_t> sums(backend->hardware()?0:size_t(last.co)*last.oh*last.ow);
#ifdef U02_STREAM_PROFILE_PHASES
            const auto alloc_end=std::chrono::steady_clock::now();
            std::cerr<<"U02_STREAM_PHASE group="<<group_id<<" elements="<<image.size()
                     <<" range_ms="<<std::chrono::duration<double,std::milli>(range_end-qbegin).count()
                     <<" quant_ms="<<std::chrono::duration<double,std::milli>(quant_end-range_end).count()
                     <<" coeff_ms="<<std::chrono::duration<double,std::milli>(coeff_end-quant_end).count()
                     <<" allocation_ms="<<std::chrono::duration<double,std::milli>(alloc_end-coeff_end).count()<<'\n';
#endif
#ifdef U02_STREAM_TRACE
            backend->stream_trace_group=unsigned(group_id);
#endif
#ifdef U02_STREAM_FP32_OUTPUT
            if(backend->hardware()) {
                // The static schedule must supply the final destination view.
                // Allocation/copy fallback would break direct output ownership.
                if(out.empty() || out.dims!=3 || out.w!=last.ow || out.h!=last.oh || out.c!=last.co ||
                   out.elemsize!=4 || out.elempack!=1 || out.cstep!=size_t(last.ow)*last.oh)
                    throw std::runtime_error("FP32 output requires the exact static schedule view");
                backend->run_stream_fp32_packet(ops,prepared_packet,quantized,static_cast<float*>(out.data));
                return 0;
            }
#endif
            const int32_t* result=backend->hardware()
                ?backend->run_stream_packet(ops,prepared_packet,quantized)
                :backend->run_stream(ops,weights,coeff,quantized,sums.data());
            const auto dbegin=std::chrono::steady_clock::now();
            out.create(last.ow,last.oh,last.co,4u,1,opt.blob_allocator);if(out.empty())return -100;
            const auto& incoming=ops.size()==1?boundary:input_scales.back();
            int output_threads=opt.num_threads;
#ifdef U02_STREAM_SMALL_TENSOR_SERIAL
            if(size_t(last.co)*last.oh*last.ow<=4096) output_threads=1;
#endif
            #pragma omp parallel for num_threads(output_threads)
            for(int ch=0;ch<last.co;++ch) {
                float* dst=out.channel(ch);
#ifdef U02_STREAM_RANGE_CACHE
                if(backend->range_cache.wants_output(group_id,ch)) {
                    float maximum=0;const uint64_t fpcr=stream_range_fpcr();
                    const bool finite=stream_dequantize_with_range(result+ch*last.oh*last.ow,dst,last.oh*last.ow,
                        incoming[last.kind?ch:0],weight_scales.back()[ch],bias.back()[ch],last.relu,maximum);
                    backend->range_cache.publish(group_id,ch,fpcr,maximum,finite);
                    range_published[ch]=finite?1:0;
                } else
#endif
                stream_dequantize_array(result+ch*last.oh*last.ow,dst,last.oh*last.ow,
                                        incoming[last.kind?ch:0],weight_scales.back()[ch],bias.back()[ch],last.relu);
            }
#ifdef U02_STREAM_RANGE_CACHE
            for(unsigned char value:range_published)if(value)backend->range_cache.published_elements+=last.oh*last.ow;
#endif
            backend->dequantize_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-dbegin).count();
#ifdef U02_STREAM_EXPORT_FP32_FIXTURE
            // Export actual per-group boundary data during complete board CPU
            // reference execution. This never loads a module or accesses PL.
            if(const char* path=std::getenv("POSE_STREAM_FP32_FIXTURE")) {
                if(backend->hardware())throw std::runtime_error("FP32 fixture requires the board integer reference");
                std::vector<StreamOutputAffine> affine(last.co);
                for(int ch=0;ch<last.co;++ch)
                    affine[ch]={incoming[last.kind?ch:0],weight_scales.back()[ch],bias.back()[ch],0};
                const auto packet=stream_pack_fp32_output(ops,weights,coeff,affine);
                const uint32_t record[6]={0x55304632u,uint32_t(packet.size()*8),uint32_t(image.size()),
                    uint32_t(last.co*last.oh*last.ow*4),uint32_t(ops.size()),uint32_t(group_id)};
                std::ofstream file(path,std::ios::binary|(backend->stream_calls==1?std::ios::trunc:std::ios::app));
                file.write(reinterpret_cast<const char*>(record),sizeof(record));
                file.write(reinterpret_cast<const char*>(packet.data()),packet.size()*8);
                file.write(reinterpret_cast<const char*>(quantized),image.size());
                for(int ch=0;ch<last.co;++ch)file.write(reinterpret_cast<const char*>(static_cast<const float*>(out.channel(ch))),size_t(last.oh)*last.ow*4);
                file.close();if(!file)throw std::runtime_error("FP32 full-network fixture export failed");
            }
#endif
#ifdef U02_STREAM_PROFILE_PHASES
            const double group_dequant_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-dbegin).count();
            std::cerr<<"U02_STREAM_GROUP group="<<group_id<<" ops="<<ops.size()
                     <<" first_kind="<<first.kind<<" input_c="<<first.ci
                     <<" input_h="<<first.h<<" input_w="<<first.w
                     <<" output_c="<<last.co<<" output_h="<<last.oh<<" output_w="<<last.ow
                     <<" wait_ms="<<backend->wait_ms-wait_before
                     <<" control_ms="<<backend->control_ms-control_before
                     <<" poll_count="<<backend->poll_count-polls_before
                     <<" dequant_ms="<<group_dequant_ms
                     <<" scope=diagnostic_includes_ddr_and_polling\n";
#endif
            return 0;
        } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return -1;}
    }
};
ncnn::Layer* create_stream(void* userdata) {return new PlStream((Backend*)userdata);}
const char* stream_model_plan_sha256() {return STREAM_MODEL_PLAN_SHA256;}
