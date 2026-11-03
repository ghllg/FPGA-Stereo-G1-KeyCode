#pragma once
#include <net.h>
#include <blob.h>
#include <layer.h>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
#include <algorithm>
#ifdef U02_STREAM_SCHEDULE_PROFILE
#include <chrono>
#include <ostream>
#include "backend.h"
#endif
#include "stream_schedule_generated.h"
#ifdef U02_STREAM_REGION_HOOK
#include "region_graph_dispatch.h"
#ifdef U02_STREAM_ASYNC_SUBMIT
#error Region hook requires synchronous graph ownership until async integration is verified
#endif
#endif
#ifdef U02_STREAM_FP32_OUTPUT
#include "backend.h"
#endif
#ifdef U02_STREAM_FUSED_RESIDUAL
#include <layer/binaryop.h>
#include <layer/relu.h>
#ifndef U02_STREAM_FUSED_LAYOUT
#error "Residual fusion requires the alias-safe layout checks"
#endif
#endif
#ifdef U02_STREAM_FUSED_LAYOUT
#include <layer/concat.h>
#include <layer/shufflechannel.h>
#endif
#ifdef U02_STREAM_FAST_NEAREST
#include <layer/interp.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#endif
#ifdef U02_LAYER_TIMING
#include <chrono>
#include <cstdio>
#endif

// Synchronous whole-network execution using the same loaded NCNN/custom layers.
// Storage lives with this runner. Split/Slice are validated CHW aliases; the
// existing PL transport retains kernel-owned coherent DMA and its barriers.
class StreamSchedule {
    ncnn::Net& net;
    std::vector<unsigned char> storage;
    std::vector<ncnn::Mat> blobs;
#ifdef U02_STREAM_REGION_HOOK
    RegionScheduleExtension* region_hook=nullptr;
#endif
#ifdef U02_STREAM_ASYNC_SUBMIT
    Backend* async_backend=nullptr;
#endif
#ifdef U02_STREAM_SCHEDULE_PROFILE
    struct LayerProfile {double elapsed=0,prepare=0,wait=0,quantize=0,control=0;uint64_t calls=0;bool executed=false;};
    std::vector<LayerProfile> layer_profiles;
    Backend* profile_backend=nullptr;
#endif
#ifdef U02_STREAM_FUSED_RESIDUAL
    struct ResidualFusion {int first=-1,second=-1,destination=-1;};
    std::vector<ResidualFusion> residual_fusions;
    std::vector<int> residual_copies;
    size_t residual_count=0,residual_direct_count=0;
    void bind_residual_fusions() {
        residual_fusions.resize(layer_count());residual_copies.assign(layer_count(),-1);
        std::vector<size_t> birth(blob_count(),layer_count()),death(blob_count(),0),consumers(blob_count(),0);
        for(size_t i=0;i<layer_count();++i) {
            const auto& s=STREAM_SCHEDULE_LAYERS[i];
            for(int b:s.outputs)birth[b]=std::min(birth[b],i);
            for(int b:s.inputs){death[b]=std::max(death[b],i);++consumers[b];}
        }
        death[STREAM_SCHEDULE_OUTPUT]=layer_count();
        for(size_t i=0;i+1<layer_count();++i) {
            const auto& add=STREAM_SCHEDULE_LAYERS[i];const auto& relu=STREAM_SCHEDULE_LAYERS[i+1];
            if(add.type!=std::string("BinaryOp") || relu.type!=std::string("ReLU") || add.inputs.size()!=2 ||
               add.outputs.size()!=1 || relu.inputs!=add.outputs || relu.outputs.size()!=1 ||
               consumers[add.outputs[0]]!=1 || add.outputs[0]==STREAM_SCHEDULE_OUTPUT)continue;
            const auto* operation=dynamic_cast<const ncnn::BinaryOp*>(net.layers()[i]);
            const auto* activation=dynamic_cast<const ncnn::ReLU*>(net.layers()[i+1]);
            if(!operation || operation->op_type!=ncnn::BinaryOp::Operation_ADD || operation->with_scalar ||
               !activation || activation->slope!=0.f)continue;
            const auto& a=blobs[add.inputs[0]];const auto& b=blobs[add.inputs[1]];
            const auto& intermediate=blobs[add.outputs[0]];const auto& final=blobs[relu.outputs[0]];
            if(a.c!=b.c || a.w!=b.w || a.h!=b.h || a.c!=intermediate.c || a.w!=intermediate.w ||
               a.h!=intermediate.h || a.c!=final.c || a.w!=final.w || a.h!=final.h)continue;
            if(overlaps(a,intermediate) || overlaps(b,intermediate))
                throw std::runtime_error("Residual intermediate overlaps live input");
            bool direct=true;
            for(size_t t=0;t<blob_count();++t)
                if(birth[t]<=i && death[t]>=i && overlaps(final,blobs[t])){direct=false;break;}
            auto& f=residual_fusions[i];f.first=add.inputs[0];f.second=add.inputs[1];
            f.destination=direct?relu.outputs[0]:add.outputs[0];
            residual_copies[i+1]=direct?-2:add.outputs[0];
            ++residual_count;if(direct)++residual_direct_count;
        }
    }
    static void residual_add_relu(const ncnn::Mat& a,const ncnn::Mat& b,ncnn::Mat& dst,int threads) {
        if(a.dims!=3 || b.dims!=3 || dst.dims!=3 || a.elemsize!=4 || b.elemsize!=4 || dst.elemsize!=4 ||
           a.elempack!=1 || b.elempack!=1 || dst.elempack!=1 || a.w!=b.w || a.h!=b.h || a.c!=b.c ||
           dst.w!=a.w || dst.h!=a.h || dst.c!=a.c || overlaps(a,dst) || overlaps(b,dst))
            throw std::runtime_error("Fused residual shape or ownership mismatch");
        #pragma omp parallel for num_threads(threads)
        for(int c=0;c<a.c;++c) {
            const float* left=a.channel(c);const float* right=b.channel(c);float* output=dst.channel(c);
            int i=0,count=a.w*a.h;
#ifdef __ARM_NEON
            const float32x4_t zero=vdupq_n_f32(0.f);
            for(;i+4<=count;i+=4)
                vst1q_f32(output+i,vmaxq_f32(vaddq_f32(vld1q_f32(left+i),vld1q_f32(right+i)),zero));
#endif
            for(;i<count;++i)output[i]=std::max(left[i]+right[i],0.f);
        }
    }
#endif
#ifdef U02_STREAM_FUSED_LAYOUT
    struct LayoutFusion {int first=-1,second=-1,destination=-1;};
    std::vector<LayoutFusion> layout_fusions;
    // -1 is an ordinary layer, -2 means its output was produced directly.
    std::vector<int> shuffle_copies;
    size_t layout_count=0,direct_layout_count=0;
    static bool overlaps(const ncnn::Mat& a,const ncnn::Mat& b) {
        const uintptr_t ab=reinterpret_cast<uintptr_t>(a.data),bb=reinterpret_cast<uintptr_t>(b.data);
        return ab<bb+size_t(b.c)*b.cstep*4 && bb<ab+size_t(a.c)*a.cstep*4;
    }
    static void concat_shuffle(const ncnn::Mat& a,const ncnn::Mat& b,ncnn::Mat& dst) {
        if(a.dims!=3 || b.dims!=3 || dst.dims!=3 || a.c!=b.c || dst.c!=a.c+b.c ||
           a.w!=b.w || a.h!=b.h || dst.w!=a.w || dst.h!=a.h ||
           a.elemsize!=4 || b.elemsize!=4 || dst.elemsize!=4 ||
           a.elempack!=1 || b.elempack!=1 || dst.elempack!=1 ||
           overlaps(a,dst) || overlaps(b,dst))
            throw std::runtime_error("Static concat/shuffle shape or overlap mismatch");
        const size_t bytes=size_t(a.w)*a.h*4;
        for(int c=0;c<a.c;++c) {
            std::memcpy(dst.channel(2*c),a.channel(c),bytes);
            std::memcpy(dst.channel(2*c+1),b.channel(c),bytes);
        }
    }
    void bind_layout_fusions() {
        layout_fusions.resize(layer_count());shuffle_copies.assign(layer_count(),-1);
        std::vector<size_t> birth(blob_count(),layer_count()),death(blob_count(),0),consumers(blob_count(),0);
        for(size_t i=0;i<layer_count();++i) {
            const auto& s=STREAM_SCHEDULE_LAYERS[i];
            for(int b:s.outputs)birth[b]=std::min(birth[b],i);
            for(int b:s.inputs) {death[b]=std::max(death[b],i);++consumers[b];}
        }
        death[STREAM_SCHEDULE_OUTPUT]=layer_count();
        for(size_t i=0;i+1<layer_count();++i) {
            const auto& cat=STREAM_SCHEDULE_LAYERS[i];const auto& sh=STREAM_SCHEDULE_LAYERS[i+1];
            if(cat.type!=std::string("Concat") || sh.type!=std::string("ShuffleChannel") ||
               cat.inputs.size()!=2 || cat.outputs.size()!=1 || sh.inputs!=cat.outputs ||
               sh.outputs.size()!=1 || consumers[cat.outputs[0]]!=1 || cat.outputs[0]==STREAM_SCHEDULE_OUTPUT)
                continue;
            const auto* concat=dynamic_cast<const ncnn::Concat*>(net.layers()[i]);
            const auto* shuffle=dynamic_cast<const ncnn::ShuffleChannel*>(net.layers()[i+1]);
            if(!concat || !shuffle || concat->axis!=0 || shuffle->group!=2 || shuffle->reverse)continue;
            const auto& a=blobs[cat.inputs[0]];const auto& b=blobs[cat.inputs[1]];
            const auto& intermediate=blobs[cat.outputs[0]];const auto& final=blobs[sh.outputs[0]];
            if(a.c!=b.c || a.w!=b.w || a.h!=b.h || intermediate.c!=2*a.c ||
               intermediate.w!=a.w || intermediate.h!=a.h || final.c!=intermediate.c ||
               final.w!=a.w || final.h!=a.h)continue;
            if(overlaps(a,intermediate) || overlaps(b,intermediate))
                throw std::runtime_error("Static concat intermediate overlaps its inputs");
            // A direct destination starts one layer earlier. Check every live
            // tensor view, including Split/Slice aliases, before writing it.
            bool direct=true;
            for(size_t t=0;t<blob_count();++t)
                if(birth[t]<=i && death[t]>=i && overlaps(final,blobs[t])) {direct=false;break;}
            auto& f=layout_fusions[i];f.first=cat.inputs[0];f.second=cat.inputs[1];
            f.destination=direct?sh.outputs[0]:cat.outputs[0];
            // The intermediate has exactly one consumer. It may carry the
            // already shuffled value until the immediately following copy.
            shuffle_copies[i+1]=direct?-2:cat.outputs[0];
            ++layout_count;if(direct)++direct_layout_count;
        }
    }
#endif
#ifdef U02_STREAM_FAST_NEAREST
    std::vector<int> nearest_scales;
    size_t nearest_count=0;
    static void nearest_repeat(const ncnn::Mat& src,ncnn::Mat& dst,int scale,int threads) {
        if(src.dims!=3 || dst.dims!=3 || src.elemsize!=4 || dst.elemsize!=4 ||
           src.elempack!=1 || dst.elempack!=1 || dst.w!=src.w*scale ||
           dst.h!=src.h*scale || dst.c!=src.c)
            throw std::runtime_error("Static nearest shape mismatch");
        const uintptr_t begin=reinterpret_cast<uintptr_t>(src.data);
        const uintptr_t end=begin+size_t(src.c)*src.cstep*4;
        const uintptr_t out_begin=reinterpret_cast<uintptr_t>(dst.data);
        const uintptr_t out_end=out_begin+size_t(dst.c)*dst.cstep*4;
        if(begin<out_end && out_begin<end)
            throw std::runtime_error("Static nearest input/output overlap");
#pragma omp parallel for num_threads(threads)
        for(int c=0;c<src.c;++c) {
            const ncnn::Mat in=src.channel(c);ncnn::Mat out=dst.channel(c);
            for(int y=0;y<src.h;++y) {
                const float* row=in.row(y);float* repeated=out.row(y*scale);
                int x=0;
#ifdef __ARM_NEON
                if(scale==2) {
                    for(;x+3<src.w;x+=4) {
                        const float32x4_t v=vld1q_f32(row+x);
                        const float32x4x2_t pairs=vzipq_f32(v,v);
                        vst1q_f32(repeated+x*2,pairs.val[0]);
                        vst1q_f32(repeated+x*2+4,pairs.val[1]);
                    }
                } else {
                    for(;x<src.w;++x) {
                        const float32x4_t v=vdupq_n_f32(row[x]);
                        vst1q_f32(repeated+x*scale,v);
                        if(scale==8)vst1q_f32(repeated+x*scale+4,v);
                    }
                }
#endif
                for(;x<src.w;++x)
                    for(int dx=0;dx<scale;++dx)repeated[x*scale+dx]=row[x];
                for(int dy=1;dy<scale;++dy)
                    std::memcpy(out.row(y*scale+dy),repeated,size_t(dst.w)*4);
            }
        }
    }
#endif
    static size_t blob_count() {return sizeof(STREAM_SCHEDULE_BLOBS)/sizeof(StreamBlobSpec);}
    static size_t layer_count() {return sizeof(STREAM_SCHEDULE_LAYERS)/sizeof(StreamLayerSpec);}
    static void copy_mat(const ncnn::Mat& src,ncnn::Mat& dst) {
        if(src.dims!=3 || src.w!=dst.w || src.h!=dst.h || src.c!=dst.c || src.elempack!=1 || src.elemsize!=4)
            throw std::runtime_error("Static schedule shape mismatch");
        if(src.data==dst.data)return;
#ifdef U02_STREAM_FUSED_LAYOUT
        if(src.cstep==size_t(src.w)*src.h && dst.cstep==size_t(dst.w)*dst.h) {
            std::memmove(dst.data,src.data,size_t(src.c)*src.w*src.h*4);return;
        }
#endif
        for(int c=0;c<src.c;++c)std::memcpy(dst.channel(c),src.channel(c),size_t(src.w)*src.h*4);
    }
public:
#ifdef U02_STREAM_REGION_HOOK
    void attach_region(RegionScheduleExtension* extension) {
        if(!extension){region_hook=nullptr;return;}
        std::vector<RegionGraphView> views;
        for(const auto& blob:blobs) {
            if(blob.dims!=3 || blob.elemsize!=4 || blob.elempack!=1)
                throw std::runtime_error("Region hook requires FP32 CHW graph views");
            views.push_back({static_cast<float*>(blob.data),unsigned(blob.c),unsigned(blob.h),unsigned(blob.w),blob.cstep});
        }
        extension->bind(STREAM_SCHEDULE_PARAM,layer_count(),views);
        region_hook=extension;
    }
#endif
    StreamSchedule(ncnn::Net& n,const char* param
#if defined(U02_STREAM_FP32_OUTPUT) || defined(U02_CALL_DIAGNOSTIC)
        ,Backend* output_backend=nullptr
#endif
        ):net(n),storage(STREAM_SCHEDULE_ARENA_BYTES+63),blobs(blob_count()) {
#ifdef U02_STREAM_ASYNC_SUBMIT
        async_backend=output_backend;
#endif
#ifdef U02_STREAM_SCHEDULE_PROFILE
        layer_profiles.resize(layer_count());
        profile_backend=output_backend;
#endif
        std::ifstream file(param,std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
        if(text!=STREAM_SCHEDULE_PARAM)
            throw std::runtime_error("Static schedule requires its exact model parameter file");
        if(net.layers().size()!=layer_count() || net.blobs().size()!=blob_count())
            throw std::runtime_error("Static schedule graph size mismatch");
        if(net.opt.use_packing_layout || net.opt.use_fp16_storage || net.opt.use_bf16_storage)
            throw std::runtime_error("Static schedule requires FP32 CHW");
        unsigned char* base=reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(storage.data())+63)&~uintptr_t(63));
#ifdef U02_STREAM_FP32_OUTPUT
        if(output_backend && output_backend->hardware()) {
            base=output_backend->stream_fp32_arena(STREAM_SCHEDULE_ARENA_BYTES);
            if(!base || reinterpret_cast<uintptr_t>(base)%64)
                throw std::runtime_error("FP32 static schedule DMA base alignment");
        }
#endif
        for(size_t i=0;i<blob_count();++i) {
            const auto& b=STREAM_SCHEDULE_BLOBS[i];
            if(net.blobs()[i].name!=b.name || b.offset+size_t(b.c)*b.h*b.w*4>STREAM_SCHEDULE_ARENA_BYTES)
                throw std::runtime_error("Static schedule blob binding mismatch");
            blobs[i]=ncnn::Mat(b.w,b.h,b.c,base+b.offset,4u,1);
            if(blobs[i].cstep!=size_t(b.w)*b.h)throw std::runtime_error("Static schedule stride mismatch");
        }
        for(size_t i=0;i<layer_count();++i) {
            const auto& s=STREAM_SCHEDULE_LAYERS[i];const auto* layer=net.layers()[i];
            if(layer->name!=s.name || layer->type!=s.type || layer->bottoms!=s.inputs || layer->tops!=s.outputs)
                throw std::runtime_error("Static schedule layer binding mismatch");
        }
#ifdef U02_STREAM_FAST_NEAREST
        nearest_scales.assign(layer_count(),0);
        for(size_t i=0;i<layer_count();++i) {
            const auto* layer=net.layers()[i];
            if(layer->type!="Interp")continue;
            const auto* interp=dynamic_cast<const ncnn::Interp*>(layer);
            if(!interp || !layer->one_blob_only || interp->resize_type!=1 ||
               interp->dynamic_target_size || !interp->size_expr.empty() ||
               interp->output_width || interp->output_height ||
               interp->width_scale!=interp->height_scale)continue;
            const float scale=interp->width_scale;
            if(scale!=2.f && scale!=4.f && scale!=8.f)continue;
            const auto& s=STREAM_SCHEDULE_LAYERS[i];
            if(s.inputs.size()!=1 || s.outputs.size()!=1)
                throw std::runtime_error("Static nearest arity mismatch");
            const auto& src=blobs[s.inputs[0]];const auto& dst=blobs[s.outputs[0]];
            if(dst.w!=src.w*int(scale) || dst.h!=src.h*int(scale) || dst.c!=src.c)
                throw std::runtime_error("Static nearest binding mismatch");
            nearest_scales[i]=int(scale);++nearest_count;
        }
#endif
#ifdef U02_STREAM_FUSED_LAYOUT
        bind_layout_fusions();
#endif
#ifdef U02_STREAM_FUSED_RESIDUAL
        bind_residual_fusions();
#endif
    }
    size_t fused_residual_pairs() const {
#ifdef U02_STREAM_FUSED_RESIDUAL
        return residual_count;
#else
        return 0;
#endif
    }
    size_t direct_residual_pairs() const {
#ifdef U02_STREAM_FUSED_RESIDUAL
        return residual_direct_count;
#else
        return 0;
#endif
    }
    size_t fused_layout_pairs() const {
#ifdef U02_STREAM_FUSED_LAYOUT
        return layout_count;
#else
        return 0;
#endif
    }
    size_t direct_layout_pairs() const {
#ifdef U02_STREAM_FUSED_LAYOUT
        return direct_layout_count;
#else
        return 0;
#endif
    }
    size_t fast_nearest_layers() const {
#ifdef U02_STREAM_FAST_NEAREST
        return nearest_count;
#else
        return 0;
#endif
    }
    void forward(const ncnn::Mat& input,ncnn::Mat& output) {
#ifdef U02_STREAM_REGION_HOOK
        if(region_hook)region_hook->begin_forward();
#endif
        copy_mat(input,blobs[STREAM_SCHEDULE_INPUT]);
        for(size_t i=0;i<layer_count();++i) {
#ifdef U02_CALL_DIAGNOSTIC
            if(profile_backend) {
                profile_backend->diagnostic_node=int(i);
                profile_backend->diagnostic_name=net.layers()[i]->name.c_str();
            }
#endif
#ifdef U02_STREAM_REGION_HOOK
            // Publish staged boundaries at their original birth nodes, even
            // when the corresponding ordinary layer is Split or Slice.
            if(region_hook && region_hook->handle_node(i))continue;
#endif
            const auto& s=STREAM_SCHEDULE_LAYERS[i];const auto* layer=net.layers()[i];
            if(layer->type=="Input" || layer->type=="Split" || layer->type=="Slice")continue;
#ifdef U02_STREAM_ASYNC_SUBMIT
            if(async_backend) {
                // Includes aliases, future output writes and the driver's
                // output guard cache line. Never rely only on graph names.
                for(int index:s.inputs) {const auto& value=blobs[index];async_backend->synchronize_span(value.data,value.c*value.cstep*4);}
                for(int index:s.outputs){const auto& value=blobs[index];async_backend->synchronize_span(value.data,value.c*value.cstep*4);}
#ifdef U02_STREAM_FUSED_LAYOUT
                if(layout_fusions[i].destination>=0) {const auto& value=blobs[layout_fusions[i].destination];async_backend->synchronize_span(value.data,value.c*value.cstep*4);}
#endif
#ifdef U02_STREAM_FUSED_RESIDUAL
                if(residual_fusions[i].destination>=0) {const auto& value=blobs[residual_fusions[i].destination];async_backend->synchronize_span(value.data,value.c*value.cstep*4);}
#endif
            }
#endif
#ifdef U02_STREAM_SCHEDULE_PROFILE
            const auto profile_begin=std::chrono::steady_clock::now();
            const double before_prepare=profile_backend?profile_backend->prepare_ms:0;
            const double before_wait=profile_backend?profile_backend->wait_ms:0;
            const double before_quantize=profile_backend?profile_backend->quantize_ms:0;
            const double before_control=profile_backend?profile_backend->control_ms:0;
            const uint64_t before_calls=profile_backend?profile_backend->calls:0;
#endif
#ifdef U02_LAYER_TIMING
            const auto layer_begin=std::chrono::steady_clock::now();
#endif
            int rc=0;
#ifdef U02_STREAM_FUSED_RESIDUAL
            if(residual_fusions[i].first>=0) {
                const auto& f=residual_fusions[i];residual_add_relu(blobs[f.first],blobs[f.second],blobs[f.destination],net.opt.num_threads);
            } else if(residual_copies[i]!=-1) {
                if(residual_copies[i]>=0)copy_mat(blobs[residual_copies[i]],blobs[s.outputs[0]]);
            } else
#endif
#ifdef U02_STREAM_FUSED_LAYOUT
            if(layout_fusions[i].first>=0) {
                const auto& f=layout_fusions[i];concat_shuffle(blobs[f.first],blobs[f.second],blobs[f.destination]);
            } else if(shuffle_copies[i]!=-1) {
                if(shuffle_copies[i]>=0)copy_mat(blobs[shuffle_copies[i]],blobs[s.outputs[0]]);
            } else
#endif
#ifdef U02_STREAM_FAST_NEAREST
            if(nearest_scales[i]) {
                nearest_repeat(blobs[s.inputs[0]],blobs[s.outputs[0]],nearest_scales[i],net.opt.num_threads);
            } else
#endif
            if(layer->one_blob_only) {
                if(s.inputs.size()!=1 || s.outputs.size()!=1)throw std::runtime_error("Static unary arity mismatch");
                ncnn::Mat out=blobs[s.outputs[0]];
                if(layer->support_inplace) {
                    copy_mat(blobs[s.inputs[0]],out);rc=layer->forward_inplace(out,net.opt);
                } else rc=layer->forward(blobs[s.inputs[0]],out,net.opt);
                if(rc)throw std::runtime_error("Static layer failed: "+layer->name);
                copy_mat(out,blobs[s.outputs[0]]);
            } else {
                std::vector<ncnn::Mat> in,out;
                for(int b:s.inputs)in.push_back(blobs[b]);
                for(int b:s.outputs)out.push_back(blobs[b]);
                rc=layer->forward(in,out,net.opt);
                if(rc || out.size()!=s.outputs.size())throw std::runtime_error("Static multi-input layer failed: "+layer->name);
                for(size_t j=0;j<out.size();++j)copy_mat(out[j],blobs[s.outputs[j]]);
            }
#ifdef U02_STREAM_SCHEDULE_PROFILE
            auto& measured=layer_profiles[i];measured.executed=true;
            measured.elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-profile_begin).count();
            measured.prepare=(profile_backend?profile_backend->prepare_ms:0)-before_prepare;
            measured.wait=(profile_backend?profile_backend->wait_ms:0)-before_wait;
            measured.quantize=(profile_backend?profile_backend->quantize_ms:0)-before_quantize;
            measured.control=(profile_backend?profile_backend->control_ms:0)-before_control;
            measured.calls=(profile_backend?profile_backend->calls:0)-before_calls;
#endif
#ifdef U02_LAYER_TIMING
            const double elapsed=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-layer_begin).count();
            // Keep graph binding types unchanged; only the emitted label differs.
            // Includes layer work and any arena copies; excludes this logging.
            const char* label=layer->type=="Convolution"?"U02Convolution":layer->type.c_str();
            std::fprintf(stderr,"%-24s %-36s %.6fms | static_fp32_arena\n",
                         label,layer->name.c_str(),elapsed);
#endif
        }
#ifdef U02_STREAM_ASYNC_SUBMIT
        if(async_backend)async_backend->finish_forward();
#endif
#ifdef U02_STREAM_REGION_HOOK
        if(region_hook)region_hook->end_forward();
#endif
        output=blobs[STREAM_SCHEDULE_OUTPUT];
    }
#ifdef U02_STREAM_SCHEDULE_PROFILE
    void write_layer_profile(std::ostream& output) const {
        output<<",\"layer_profile_scope\":\"last_timed_complete_forward_buffered_no_per_layer_logging\",\"layer_profiles\":[";
        bool first=true;
        for(size_t i=0;i<layer_count();++i) {
            const auto& measured=layer_profiles[i];if(!measured.executed)continue;
            const auto* layer=net.layers()[i];const auto& spec=STREAM_SCHEDULE_LAYERS[i];
            output<<(first?"":",")<<"{\"index\":"<<i<<",\"name\":\""<<layer->name.c_str()
                <<"\",\"type\":\""<<layer->type.c_str()<<"\",\"elapsed_ms\":"<<measured.elapsed
                <<",\"prepare_ms\":"<<measured.prepare<<",\"pl_wait_ms\":"<<measured.wait
                <<",\"quantize_ms\":"<<measured.quantize<<",\"control_ms\":"<<measured.control
                <<",\"pl_calls\":"<<measured.calls<<",\"output_shape\":[";
            if(spec.outputs.size()==1) {
                const auto& shape=STREAM_SCHEDULE_BLOBS[spec.outputs[0]];
                output<<shape.c<<","<<shape.h<<","<<shape.w;
            }
            output<<"]}";first=false;
        }
        output<<"]";
    }
#endif
};
