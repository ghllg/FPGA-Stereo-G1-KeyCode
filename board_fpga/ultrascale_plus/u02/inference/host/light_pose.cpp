#include "pl_conv_adapter.h"
#include "pl_depthwise.h"
#include "pl_pair.h"
#ifdef U02_STEREO_CONTINUOUS
#include <csignal>
#include <thread>
static volatile std::sig_atomic_t stereo_stop_requested=0;
static void request_stereo_stop(int){stereo_stop_requested=1;}
#endif
#ifdef U02_IMAGE_PIPELINE
#include "image_preprocess.h"
#endif
#ifdef U02_CAMERA_PIPELINE
#include "camera_capture.h"
#ifdef U02_STEREO_PIPELINE
#include "stereo_capture.h"
#ifdef U02_STEREO_NETWORK
#include "paired_pose_feed.h"
#endif
#endif
#endif
#ifdef U02_STREAM_SCHEDULE
#include "stream_schedule.h"
#endif
#ifdef U02_STREAM_MODEL
#include "pl_stream.h"
#endif
#ifdef U02_STREAM_TRACE
#include "stream_trace.h"
#endif

struct ForwardProfile {
    uint64_t calls=0, ordinary=0, depthwise=0, macs=0, streams=0;
    uint64_t prepared_streams=0,direct_stream_inputs=0;
    uint64_t resident_packet_calls=0,resident_prefix_copy_bytes=0;
    double prepare=0, wait=0, quantize=0, dequantize=0, readback=0, control=0;
    explicit ForwardProfile(const Backend* b) {
        if(!b) return;
        calls=b->calls;ordinary=b->convolution_calls;depthwise=b->depthwise_calls;macs=b->macs;
        streams=b->stream_calls;
        prepared_streams=b->stream_prepared_packet_calls;direct_stream_inputs=b->stream_direct_input_calls;
        resident_packet_calls=b->stream_resident_packet_calls;
        resident_prefix_copy_bytes=b->stream_resident_prefix_copy_bytes;
        prepare=b->prepare_ms;wait=b->wait_ms;quantize=b->quantize_ms;
        dequantize=b->dequantize_ms;readback=b->readback_ms;control=b->control_ms;
    }
};

// Reused normalized CHW input; the entire forward graph executes on the board.
// ROI describes the exact affine crop in original source-image coordinates.
int main(int argc,char** argv) {
    try {
        if(argc!=12) throw std::runtime_error("Usage: light_pose cpu|csim|pl|stream-csim|pl-stream PARAM BIN INPUT.f32 OUTPUT.json SOURCE_W SOURCE_H ROI_X ROI_Y ROI_W ROI_H");
        const std::string mode=argv[1];
        const bool streaming=(mode=="stream-csim" || mode=="pl-stream"
#if defined(U02_STREAM_REGION_HOOK) && defined(U02_STREAM_FP32_OUTPUT)
            || mode=="pl-region"
#endif
            );
        if(mode!="cpu" && mode!="csim" && mode!="pl" && !streaming) throw std::runtime_error("Unknown backend");
#ifdef U02_STEREO_PROCESSING_DIAGNOSTIC
        if(mode!="cpu" && mode!="pl-stream")throw std::runtime_error("Processing diagnostic requires cpu or pl-stream");
#endif
        const int source_w=std::stoi(argv[6]),source_h=std::stoi(argv[7]);
        const double rx=std::stod(argv[8]),ry=std::stod(argv[9]),rw=std::stod(argv[10]),rh=std::stod(argv[11]);
        if(source_w<1 || source_h<1 || !std::isfinite(rx+ry+rw+rh) || rw<=0 || rh<=0)
            throw std::runtime_error("Invalid source/ROI geometry");
        const int threads=std::getenv("POSE_THREADS")?std::stoi(std::getenv("POSE_THREADS")):1;
        if(threads<1 || threads>4) throw std::runtime_error("Use 1 to 4 CPU threads");
#ifndef _OPENMP
        if(threads!=1) throw std::runtime_error("OpenMP build required for multiple threads");
#endif
        std::unique_ptr<Backend> backend;
        std::unique_ptr<Backend> stem_reference;
        if(mode!="cpu") backend=make_backend(mode);
        if(streaming) stem_reference=make_backend((mode=="pl-stream" || mode=="pl-region")?"arm-stem":"csim");
        if(stem_reference) stem_reference->cpu_threads=threads;
        const bool calibrate=std::getenv("POSE_CALIBRATION") && std::string(std::getenv("POSE_CALIBRATION"))=="1";
        if(calibrate && !backend) throw std::runtime_error("Calibration requires custom convolution backend");
        if(backend) backend->collect_calibration=calibrate;
        ncnn::Net net;net.opt.num_threads=threads;net.opt.use_vulkan_compute=false;
#ifdef U02_STREAM_TRACE
        if(mode!="csim" && mode!="stream-csim")throw std::runtime_error("Boundary trace supports software references only");
        net.opt.lightmode=false;
#endif
        // Use NCNN's FP32 SIMD packing for the CPU baseline. Custom PL layers
        // presently require scalar CHW tensors and will account for that overhead.
        net.opt.use_packing_layout=(mode=="cpu");net.opt.use_fp16_storage=false;
        net.opt.use_fp16_arithmetic=false;net.opt.use_bf16_storage=false;
        if(backend && !streaming) net.register_custom_layer("Convolution",create_conv,nullptr,backend.get());
        if(streaming) net.register_custom_layer("Convolution",create_conv,nullptr,stem_reference.get());
        if(backend) net.register_custom_layer("LightPwDw",create_pair,nullptr,backend.get());
#ifdef U02_STREAM_MODEL
        if(backend) net.register_custom_layer("U02Stream",create_stream,nullptr,backend.get());
#endif
        const bool use_dw=std::getenv("LIGHT_PL_DW") && std::string(std::getenv("LIGHT_PL_DW"))=="1";
        if(backend && use_dw && !streaming) net.register_custom_layer("ConvolutionDepthWise",create_depthwise,nullptr,backend.get());
        if(net.load_param(argv[2]) || net.load_model(argv[3])) throw std::runtime_error("Model load failed");
#if defined(U02_LAYER_TIMING) && !defined(U02_STREAM_SCHEDULE)
        // NCNN's diagnostic formatter casts these names to its built-in
        // convolution classes. Our registered adapters derive from Layer.
        // Rename only diagnostic labels after loading/pipeline creation.
        if(backend) for(ncnn::Layer* layer:net.mutable_layers()) {
            if(layer->type=="Convolution" || (use_dw && !streaming && layer->type=="ConvolutionDepthWise"))
                layer->type="U02"+layer->type;
        }
#endif
        ncnn::Mat input(96,128,3);
#ifdef U02_IMAGE_PIPELINE
        const std::string image_flag=std::getenv("POSE_IMAGE_INPUT")?std::getenv("POSE_IMAGE_INPUT"):"1";
        if(image_flag!="0" && image_flag!="1")throw std::runtime_error("POSE_IMAGE_INPUT must be 0 or 1");
        const bool image_input=image_flag=="1";
        std::vector<uint8_t> encoded;
#ifdef U02_CAMERA_PIPELINE
        if(!image_input)throw std::runtime_error("Camera pipeline requires JPEG input");
        if(mode!="cpu" && mode!="stream-csim" && mode!="pl-stream")
            throw std::runtime_error("Camera pipeline requires a complete CPU or streaming backend");
#ifdef U02_STEREO_PIPELINE
        std::unique_ptr<StereoCapture> camera(new StereoCapture(argv[4],source_w,source_h,rx,ry,rw,rh));
#else
        std::unique_ptr<CameraCapture> camera(new CameraCapture(argv[4],source_w,source_h));
#endif
#endif
        if(image_input) {
            if(input.cstep!=96*128)throw std::runtime_error("Expected contiguous CHW input");
#ifndef U02_CAMERA_PIPELINE
            encoded=read_pose_image(argv[4]);
#endif
        } else
#endif
        {
        std::ifstream f(argv[4],std::ios::binary);
        for(int c=0;c<3;++c) if(!f.read((char*)(float*)input.channel(c),96*128*4)) throw std::runtime_error("Truncated normalized input");
        if(f.peek()!=EOF) throw std::runtime_error("Unexpected input length");
        for(int c=0;c<3;++c) for(int i=0;i<96*128;++i)
            if(!std::isfinite(((const float*)input.channel(c))[i])) throw std::runtime_error("Nonfinite input");
        }
#ifdef U02_STEREO_CONTINUOUS
        int repeats=2000000000;
        const int run_seconds=std::getenv("POSE_RUN_SECONDS")?std::stoi(std::getenv("POSE_RUN_SECONDS")):300;
        if(run_seconds<2 || run_seconds>3600)throw std::runtime_error("POSE_RUN_SECONDS must be 2..3600");
        std::signal(SIGINT,request_stereo_stop);std::signal(SIGTERM,request_stereo_stop);
        uint64_t total_forwards=0,completed_pairs=0,total_pl_calls=0;
#ifndef U02_STEREO_NETWORK
        uint64_t non_network_expired_pairs=0;
#endif
        bool delay_injected=false;
#else
        const int repeats=std::getenv("POSE_REPEAT")?std::stoi(std::getenv("POSE_REPEAT")):1;
#endif
        const int warmup=std::getenv("POSE_WARMUP")?std::stoi(std::getenv("POSE_WARMUP")):0;
        if(repeats<1 ||
#ifndef U02_STEREO_CONTINUOUS
           repeats>1000 ||
#endif
           warmup<0 || warmup>100)
            throw std::runtime_error("Invalid inference repetitions");
#ifdef U02_STEREO_PIPELINE
        if(warmup!=0 || repeats%2)throw std::runtime_error("Stereo requires zero warmup and an even number of image forwards");
#endif
#ifdef U02_STREAM_TRACE
        if(repeats!=1 || warmup!=0)throw std::runtime_error("Boundary trace is diagnostic, not a benchmark");
#endif
        if(calibrate && (repeats!=1 || warmup!=0))
            throw std::runtime_error("Calibration is diagnostic: require POSE_REPEAT=1 and POSE_WARMUP=0");
        if(std::getenv("POSE_STREAM_FIXTURE") && ((mode!="csim" && mode!="stream-csim") || repeats!=1 || warmup!=0))
            throw std::runtime_error("Fixture capture requires one board integer-reference forward");
        ncnn::Mat out;
#ifdef U02_STREAM_SCHEDULE
        std::unique_ptr<StreamSchedule> schedule;
        if(streaming) schedule.reset(new StreamSchedule(net,argv[2]
#if defined(U02_STREAM_FP32_OUTPUT) || defined(U02_CALL_DIAGNOSTIC)
            ,backend.get()
#endif
            ));
#if defined(U02_STREAM_REGION_HOOK) && defined(U02_STREAM_FP32_OUTPUT)
        if(mode=="pl-region") {
            auto* extension=backend->region_extension();
            if(!schedule || !extension)throw std::runtime_error("Fused backend did not provide its complete-graph extension");
            schedule->attach_region(extension);
        }
#endif
#endif
        std::vector<double> times;
        std::vector<ForwardProfile> profiles;
#ifdef U02_IMAGE_PIPELINE
        struct ImageProfile {double decode,roi,forward,pose,total;};
        std::vector<ImageProfile> image_profiles;
        auto decode_pipeline_heatmap=[&](const ncnn::Mat& heat) {
            if(heat.dims!=3 || heat.w!=24 || heat.h!=32 || heat.c!=17 || heat.elempack!=1)
                throw std::runtime_error("Expected COCO17 heatmap 17x32x24");
            std::vector<float> points(51);
            for(int c=0;c<17;++c) {
                const float* h=heat.channel(c);int best=0;
                for(int i=0;i<768;++i) {
                    if(!std::isfinite(h[i]))throw std::runtime_error("Nonfinite heatmap");
                    if(h[i]>h[best])best=i;
                }
                float x=best%24,y=best/24;
                if(x>0 && x<23 && y>0 && y<31) {
                    const float dx=h[best+1]-h[best-1],dy=h[best+24]-h[best-24];
                    x+=dx>0?.25f:dx<0?-.25f:0;y+=dy>0?.25f:dy<0?-.25f:0;
                }
                points[c*3]=float(rx+x*rw/24.0);points[c*3+1]=float(ry+y*rh/32.0);points[c*3+2]=h[best];
            }
            return points;
        };
        std::vector<float> pipeline_points;
#ifdef U02_STEREO_NETWORK
        if(mode!="pl-stream")throw std::runtime_error("Paired network entry requires actual whole-network FPGA backend");
        PairedPoseFeed pair_feed;
#endif
#ifdef U02_CAMERA_PIPELINE
        struct CameraProfile {CameraFrame frame;std::vector<float> points;};
        std::vector<CameraProfile> camera_profiles;
        std::chrono::steady_clock::time_point camera_timed_start,camera_timed_end;
#endif
#endif
        for(int iteration=-warmup;iteration<repeats;++iteration) {
#ifdef U02_STEREO_CONTINUOUS
            if(total_forwards && camera->at_pair_boundary() && (stereo_stop_requested ||
                std::chrono::duration<double>(std::chrono::steady_clock::now()-camera_timed_start).count()>=run_seconds))break;
#endif
            if(backend) backend->reset_forward_metrics();
            if(stem_reference) stem_reference->reset_forward_metrics();
#ifdef U02_IMAGE_PIPELINE
            const auto pipeline_start=std::chrono::steady_clock::now();
#ifdef U02_CAMERA_PIPELINE
            if(iteration==0)camera_timed_start=pipeline_start;
            CameraFrame camera_frame;
#ifdef U02_STEREO_CONTINUOUS
            try{camera_frame=camera->next(encoded);}
            catch(const StereoPairExpired&) {
                camera->discard_expired_pending();
#ifdef U02_STEREO_NETWORK
                pair_feed.discard_expired_pair();
#else
                ++non_network_expired_pairs;
#endif
                // The left computation happened, but the half-pair is never
                // published as a valid paired result or kept in its window.
                camera_profiles.pop_back();image_profiles.pop_back();times.pop_back();profiles.pop_back();
                continue;
            }
#else
            camera_frame=camera->next(encoded);
#endif
#endif
            PoseImageTimes preparation;
#ifdef U02_STEREO_PIPELINE
            preparation=camera->prepared_input((float*)input.data);
#else
            if(image_input)preparation=prepare_pose_image(encoded,source_w,source_h,rx,ry,rw,rh,(float*)input.data);
#endif
#endif
            const auto begin=std::chrono::steady_clock::now();
#ifdef U02_STEREO_PIPELINE
            camera_frame.infer_start_ns=stereo_monotonic_ns();
#endif
#ifdef U02_STREAM_SCHEDULE
            if(schedule) schedule->forward(input,out);
            else
#endif
            {
                auto ex=net.create_extractor();
                if(ex.input("image",input) || ex.extract("conv2d_441.tmp_1",out))
                    throw std::runtime_error("Inference failed");
#ifdef U02_STREAM_TRACE
                write_stream_trace(ex,argv[5]);
#endif
            }
            const auto finish=std::chrono::steady_clock::now();
#ifdef U02_STEREO_PIPELINE
            camera_frame.infer_end_ns=stereo_monotonic_ns();
#ifndef U02_STEREO_CONTINUOUS
            std::cerr<<"STEREO_FORWARD camera="<<camera_frame.camera_id<<" pair="<<camera_frame.pair_id
                     <<" source_us="<<camera_frame.timestamp_us<<" dequeue_ns="<<camera_frame.dequeue_monotonic_ns
                     <<" decode_ms="<<preparation.decode_ms<<" infer_ns="
                     <<camera_frame.infer_end_ns-camera_frame.infer_start_ns<<"\n";
#endif
#endif
            if(stem_reference && (stem_reference->calls!=1 || stem_reference->macs!=2654208))
                throw std::runtime_error("Streaming mode requires exactly one known ARM integer stem");
#ifdef U02_IMAGE_PIPELINE
            const auto pose_start=std::chrono::steady_clock::now();
            pipeline_points=decode_pipeline_heatmap(out);
#ifdef U02_STEREO_NETWORK
            pair_feed.submit(camera_frame,pipeline_points,backend->calls,source_w,source_h,rx,ry,rw,rh);
#endif
            const auto pipeline_finish=std::chrono::steady_clock::now();
#ifdef U02_CAMERA_PIPELINE
            if(iteration>=0)camera_profiles.push_back({camera_frame,pipeline_points});
            camera_timed_end=pipeline_finish;
#endif
            if(iteration>=0) image_profiles.push_back({preparation.decode_ms,preparation.preprocess_ms,
                std::chrono::duration<double,std::milli>(finish-begin).count(),
                std::chrono::duration<double,std::milli>(pipeline_finish-pose_start).count(),
                std::chrono::duration<double,std::milli>(pipeline_finish-pipeline_start).count()});
#endif
            if(iteration>=0) {
                times.push_back(std::chrono::duration<double,std::milli>(finish-begin).count());
                profiles.emplace_back(backend.get());
            }
#ifdef U02_STEREO_CONTINUOUS
            ++total_forwards;total_pl_calls+=(backend?backend->calls:0);
            if(camera_frame.camera_id)++completed_pairs;
            if(camera_profiles.size()>128) {
                camera_profiles.erase(camera_profiles.begin(),camera_profiles.begin()+2);
                image_profiles.erase(image_profiles.begin(),image_profiles.begin()+2);
                times.erase(times.begin(),times.begin()+2);profiles.erase(profiles.begin(),profiles.begin()+2);
            }
            if(!camera_frame.camera_id && !delay_injected && std::getenv("POSE_TEST_DELAY_LEFT_MS")) {
                const int delay=std::stoi(std::getenv("POSE_TEST_DELAY_LEFT_MS"));
                if(delay!=600)throw std::runtime_error("Only the explicit 600ms expiry recovery diagnostic is allowed");
                delay_injected=true;std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            }
#endif
        }
#ifdef U02_STEREO_CONTINUOUS
        repeats=int(total_forwards);
        if(!completed_pairs)throw std::runtime_error("No complete pair produced during continuous run");
#endif
#ifdef U02_CAMERA_PIPELINE
        camera->stop();
#endif
        const auto inferred=std::chrono::steady_clock::now();
        if(out.dims!=3 || out.w!=24 || out.h!=32 || out.c!=17 || out.elempack!=1)
            throw std::runtime_error("Expected COCO17 heatmap 17x32x24");
        std::vector<float> xy(51);int valid=0;
        for(int c=0;c<17;++c) {
            const float* h=out.channel(c);int best=0;
            for(int i=0;i<768;++i) {
                if(!std::isfinite(h[i])) throw std::runtime_error("Nonfinite heatmap");
                if(h[i]>h[best]) best=i;
            }
            float x=best%24,y=best/24;
            if(x>0 && x<23 && y>0 && y<31) {
                const float dx=h[best+1]-h[best-1],dy=h[best+24]-h[best-24];
                x+=dx>0?.25f:dx<0?-.25f:0;y+=dy>0?.25f:dy<0?-.25f:0;
            }
            xy[c*3]=float(rx+x*rw/24.0);xy[c*3+1]=float(ry+y*rh/32.0);xy[c*3+2]=h[best];
            if(h[best]>=.2f) ++valid;
        }
        const auto ended=std::chrono::steady_clock::now();
        const double ms=times.back();
        std::vector<double> sorted=times;std::sort(sorted.begin(),sorted.end());
        const double median=(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])*0.5;
        const double p95=sorted[size_t(std::ceil(sorted.size()*0.95))-1];
        std::ofstream result(argv[5]);if(!result) throw std::runtime_error("Cannot write output");
#ifdef U02_STEREO_CONTINUOUS
        result<<"{\"continuous_run\":true,\"history_limit_images\":128,\"history_scope\":\"last_complete_pairs_max_128_images\""
              <<",\"inference_statistics_scope\":\"retained_window_only\",\"total_image_forwards\":"<<total_forwards
              <<",\"completed_pairs\":"<<completed_pairs<<",\"total_pl_calls\":"<<total_pl_calls
              <<",\"expired_pair_drops\":"
#ifdef U02_STEREO_NETWORK
              <<pair_feed.expired()
#else
              <<non_network_expired_pairs
#endif
              <<",\"requested_run_seconds\":"<<run_seconds
              <<",\"stop_signal_received\":"<<(stereo_stop_requested?"true":"false")<<",";
#else
        result<<"{";
#endif
        result<<std::setprecision(9)<<"\"schema\":\"light_pose_v1\",\"model\":\"PP-TinyPose-128x96\",\"backend\":\""<<mode
          <<"\",\"scope\":\"single_image_forward_and_heatmap_decode\",\"coordinate_space\":\"source_image_pixels\",\"keypoint_order\":\"COCO17\",\"source_size\":["<<source_w<<","<<source_h
          <<"],\"roi_xywh\":["<<rx<<","<<ry<<","<<rw<<","<<rh<<"],\"infer_ms\":"<<ms
          <<",\"warmup\":"<<warmup<<",\"repeats\":"<<repeats<<",\"infer_median_ms\":"<<median<<",\"infer_p95_ms\":"<<p95
          <<",\"forward_times_ms\":[";
        for(size_t i=0;i<times.size();++i)result<<(i?",":"")<<times[i];
        result<<"],\"backend_counter_scope\":\"last_timed_forward\",\"forward_profiles\":[";
        for(size_t i=0;i<profiles.size();++i) {
            const auto& p=profiles[i];
            result<<(i?",":"")<<"{\"calls\":"<<p.calls<<",\"ordinary\":"<<p.ordinary
              <<",\"depthwise\":"<<p.depthwise<<",\"macs\":"<<p.macs<<",\"stream_dispatches\":"<<p.streams
              <<",\"stream_prepared_packet_calls\":"<<p.prepared_streams<<",\"stream_direct_input_calls\":"<<p.direct_stream_inputs
              <<",\"prepare_ms\":"<<p.prepare<<",\"pl_wait_ms\":"<<p.wait
              <<",\"quantize_ms\":"<<p.quantize<<",\"dequantize_ms\":"<<p.dequantize
              <<",\"readback_ms\":"<<p.readback<<",\"control_ms\":"<<p.control
#ifdef U02_STREAM_RESIDENT_PACKETS
              <<",\"stream_resident_packet_calls\":"<<p.resident_packet_calls
              <<",\"stream_resident_prefix_copy_bytes\":"<<p.resident_prefix_copy_bytes
#endif
              <<"}";
        }
        result<<"]"
#ifdef U02_CALL_DIAGNOSTIC
        ;if(backend)backend->write_call_diagnostics(result);result
#endif
#ifdef U02_STREAM_SCHEDULE
#ifdef U02_STREAM_SCHEDULE_PROFILE
        ;if(schedule)schedule->write_layer_profile(result);result
#endif
          <<",\"execution_schedule\":\""<<(schedule?"static_fp32_arena":"ncnn_extractor")<<"\""
          <<",\"scheduled_arena_bytes\":"<<(schedule?STREAM_SCHEDULE_ARENA_BYTES:0)
          <<",\"stream_fast_nearest_layers\":"<<(schedule?schedule->fast_nearest_layers():0)
          <<",\"stream_fused_layout_pairs\":"<<(schedule?schedule->fused_layout_pairs():0)
          <<",\"stream_direct_layout_pairs\":"<<(schedule?schedule->direct_layout_pairs():0)
          <<",\"stream_fused_residual_pairs\":"<<(schedule?schedule->fused_residual_pairs():0)
          <<",\"stream_direct_residual_pairs\":"<<(schedule?schedule->direct_residual_pairs():0)
#else
          <<",\"execution_schedule\":\"ncnn_extractor\",\"scheduled_arena_bytes\":0"
#endif
          <<",\"decode_ms\":"<<std::chrono::duration<double,std::milli>(ended-inferred).count()
          <<",\"cpu_threads\":"<<threads<<",\"pl_acceleration\":"<<(backend && backend->hardware() && backend->calls?"true":"false")
          <<",\"ncnn_packing_layout\":"<<(net.opt.use_packing_layout?"true":"false")
#ifdef U02_STREAM_FP32_OUTPUT
          <<",\"stream_output_read_in_dequantize\":false"
          <<",\"stream_output_format\":\"fp32\",\"stream_candidate\":\""<<(mode=="pl-region"?"region_dual":"fp32_direct")<<"\""
#ifdef U02_STREAM_ASYNC_SUBMIT
          <<",\"submission_mode\":\"one_pending_worker_span_guarded\",\"async_submissions\":"<<(backend?backend->async_submissions:0)
          <<",\"async_span_waits\":"<<(backend?backend->async_span_waits:0)<<",\"async_heap_inputs\":"<<(backend?backend->async_heap_inputs:0)
#endif
#ifdef U02_STREAM_DW_CHANNEL_LOCAL
          <<",\"boundary_quantization\":\"dw_channel_local\""
#endif
          <<",\"stream_packet_version\":"<<((mode=="pl-stream" || mode=="pl-region")?2:1)
          <<",\"stream_device_abi\":"<<(mode=="pl-region"?6:(mode=="pl-stream"?4:0))
          <<",\"stream_output_direct_dma\":"<<(mode=="pl-stream"?"true":"false")
#else
          <<",\"stream_output_read_in_dequantize\":"<<(mode=="pl-stream"?"true":"false")
#ifdef U02_STREAM_DW_CHANNEL_LOCAL
          <<",\"boundary_quantization\":\"dw_channel_local\""
#endif
#if defined(U02_STEREO_PROCESSING_DIAGNOSTIC)
          <<",\"processing_diagnostic\":true,\"network_sending_included\":false"
          <<",\"stream_candidate\":\""<<(mode=="cpu"?"arm_ncnn_fp32":"ingress_int32_resident")<<"\""
          <<",\"stream_device_abi\":"<<(mode=="cpu"?0:3)<<",\"stream_packet_version\":"<<(mode=="cpu"?0:1)
#elif defined(U02_FORMAL_STEREO_ABI3)
          <<",\"stream_output_format\":\"int32\",\"stream_candidate\":\"ingress_int32_resident\""
          <<",\"stream_device_abi\":3,\"stream_packet_version\":1"
#endif
#endif
          <<",\"ordinary_conv_calls\":"<<(backend?backend->convolution_calls:0)<<",\"depthwise_calls\":"<<(backend?backend->depthwise_calls:0)<<",\"pl_macs\":"<<(backend?backend->macs:0)
          <<",\"fused_pair_calls\":"<<(backend?backend->fused_pair_calls:0)<<",\"backend_dispatches\":"<<(backend?backend->calls:0)
          <<",\"prepare_ms\":"<<(backend?backend->prepare_ms:0)<<",\"pl_wait_ms\":"<<(backend?backend->wait_ms:0)
          <<",\"quantize_ms\":"<<(backend?backend->quantize_ms:0)<<",\"readback_ms\":"<<(backend?backend->readback_ms:0)
          <<",\"dequantize_ms\":"<<(backend?backend->dequantize_ms:0)
          <<",\"fine_profile\":{\"enabled\":"<<(backend && backend->fine_profile?"true":"false")
          <<",\"input_memcpy_ms\":"<<(backend?backend->input_memcpy_ms:0)<<",\"weight_memcpy_ms\":"<<(backend?backend->weight_memcpy_ms:0)
          <<",\"output_memcpy_ms\":"<<(backend?backend->output_memcpy_ms:0)<<",\"metadata_memcpy_ms\":"<<(backend?backend->metadata_memcpy_ms:0)
          <<",\"sync_cpu_ms\":"<<(backend?backend->sync_cpu_ms:0)<<",\"sync_device_ms\":"<<(backend?backend->sync_device_ms:0)
          <<",\"initial_sync_cpu_ms_excluded\":"<<(backend?backend->initial_sync_cpu_ms:0)
          <<",\"control_ms\":"<<(backend?backend->control_ms:0)<<",\"poll_count\":"<<(backend?backend->poll_count:0)
          <<",\"sync_cpu_calls\":"<<(backend?backend->sync_cpu_calls:0)<<",\"sync_device_calls\":"<<(backend?backend->sync_device_calls:0)
          <<",\"sync_cpu_bytes\":"<<(backend?backend->sync_cpu_bytes:0)<<",\"sync_device_bytes\":"<<(backend?backend->sync_device_bytes:0)
          <<"},\"valid_joints\":"<<valid<<",\"keypoints\":[";
        for(int c=0;c<17;++c) result<<(c?",":"")<<"["<<xy[c*3]<<","<<xy[c*3+1]<<","<<xy[c*3+2]<<"]";
        result<<"],\"stream_dispatches\":"<<(backend?backend->stream_calls:0)
              <<",\"calibration_enabled\":"<<(calibrate?"true":"false");
#ifdef U02_STREAM_RESIDENT_PACKETS
        result<<",\"stream_resident_packet_calls\":"<<(backend?backend->stream_resident_packet_calls:0)
              <<",\"stream_resident_prefix_copy_bytes\":"<<(backend?backend->stream_resident_prefix_copy_bytes:0)
              <<",\"stream_resident_model_packets\":"<<(backend?backend->stream_resident_model_packets:0)
              <<",\"stream_resident_model_bytes\":"<<(backend?backend->stream_resident_model_bytes:0)
              <<",\"stream_resident_pool_bytes\":"<<(backend?backend->stream_resident_pool_bytes:0)
              <<",\"stream_packet_storage\":\""<<(backend && backend->stream_resident_model_packets?"kernel_owned_ps_ddr":"none")<<"\""
              <<",\"stream_pl_weight_reads_unchanged\":"<<(backend && backend->stream_resident_model_packets?"true":"false");
#endif
        if(streaming) result<<",\"stem_backend\":\"arm_integer\",\"stem_macs\":2654208";
#ifdef U02_ARM_STEM_SMLAL
        if(streaming)result<<",\"stem_variant\":\"int16_smlal_int32_accumulate\"";
#endif
#ifdef U02_STEM_PHASE_PROFILE
        if(streaming && stem_reference) {
            result<<",\"stem_phase_scope\":\"last_timed_forward; custom_conv includes all subordinate phases and allocation\""
                  <<",\"stem_phases\":{\"total_ms\":"<<stem_reference->custom_conv_ms
                  <<",\"quantize_ms\":"<<stem_reference->quantize_ms
                  <<",\"im2col_ms\":"<<stem_reference->im2col_ms
                  <<",\"dequantize_ms\":"<<stem_reference->dequantize_ms<<"}";
        }
#endif
#ifdef U02_STREAM_MODEL
        if(backend && backend->stream_calls) result<<",\"stream_plan_sha256\":\""<<stream_model_plan_sha256()<<'"';
#endif
        if(calibrate) {
            result<<",\"calibration\":[";
            for(size_t i=0;i<backend->calibration.size();++i) {
                if(i)result<<',';
                backend->calibration[i].write(result);
            }
            result<<']';
        }
#ifdef U02_IMAGE_PIPELINE
        if(xy!=pipeline_points)throw std::runtime_error("Pipeline and output keypoint decode differ");
        std::vector<double> pipeline_times;
        for(const auto& p:image_profiles)pipeline_times.push_back(p.total);
        std::sort(pipeline_times.begin(),pipeline_times.end());
#ifdef U02_CAMERA_PIPELINE
        result<<",\"input_kind\":\"live_camera_mjpeg\""
              <<",\"pipeline_scope\":\"camera_wait_and_copy_jpeg_decode_roi_normalize_complete_forward_coco17_decode_excludes_model_loading_and_output_serialization\"";
#else
        result<<",\"input_kind\":\""<<(image_input?"encoded_image":"normalized_f32")<<"\""
              <<",\"pipeline_scope\":\""<<(image_input?
                  "cached_encoded_image_decode_roi_normalize_complete_forward_coco17_decode_excludes_file_IO_model_loading_capture_and_output_serialization":
                  "normalized_chw_complete_forward_coco17_decode_excludes_file_IO_model_loading_and_output_serialization")<<"\"";
#endif
        result<<",\"model_load_count\":1,\"pipeline_frame_count\":"<<image_profiles.size()
              <<",\"pipeline_median_ms\":"<<(pipeline_times[(pipeline_times.size()-1)/2]+pipeline_times[pipeline_times.size()/2])*.5
              <<",\"pipeline_p95_ms\":"<<pipeline_times[size_t(std::ceil(pipeline_times.size()*.95))-1]
              <<",\"pipeline_profiles\":[";
        for(size_t i=0;i<image_profiles.size();++i) {
            const auto& p=image_profiles[i];
            result<<(i?",":"")<<"{\"image_decode_ms\":"<<p.decode<<",\"roi_normalize_ms\":"<<p.roi
                <<",\"forward_ms\":"<<p.forward<<",\"heatmap_decode_ms\":"<<p.pose<<",\"total_ms\":"<<p.total<<"}";
        }
        result<<"]";
#ifdef U02_CAMERA_PIPELINE
        const double camera_elapsed=std::chrono::duration<double>(camera_timed_end-camera_timed_start).count();
#ifdef U02_FORMAL_STEREO_ABI3
        camera->write_timing_diagnostic(result);
#endif
#ifdef U02_STEREO_NETWORK
        pair_feed.flush();
        result<<",\"paired_network_frames_sent\":"<<pair_feed.sent()<<",\"paired_network_frames_dropped\":"<<pair_feed.dropped();
#endif
        result<<",\"camera_elapsed_s\":"<<camera_elapsed
#ifdef U02_STEREO_CONTINUOUS
              <<",\"observed_processed_fps\":"<<total_forwards/camera_elapsed
              <<",\"observed_completed_pairs_per_s\":"<<completed_pairs/camera_elapsed
#else
              <<",\"observed_processed_fps\":"<<camera_profiles.size()/camera_elapsed
#endif
              <<",\"camera_requested_fps\":30,\"exposure_timestamp_verified\":false,\"camera_frames\":[";
        for(size_t i=0;i<camera_profiles.size();++i) {
            const auto& p=camera_profiles[i];
            result<<(i?",":"")<<"{\"sequence\":"<<p.frame.sequence<<",\"timestamp_us\":"<<p.frame.timestamp_us
                  <<",\"flags\":"<<p.frame.flags<<",\"capture_ms\":"<<p.frame.capture_ms
                  <<",\"discarded_ready_frames\":"<<p.frame.discarded_ready_frames;
#ifdef U02_STEREO_PIPELINE
            result<<",\"camera_id\":\""<<(p.frame.camera_id?"right":"left")<<"\",\"pair_id\":"<<p.frame.pair_id
                  <<",\"frame_id\":"<<p.frame.sequence<<",\"timestamp_source\":\"v4l2_monotonic_unverified_exposure\""
                  <<",\"receive_monotonic_ns\":"<<p.frame.receive_monotonic_ns
                  <<",\"dequeue_monotonic_ns\":"<<p.frame.dequeue_monotonic_ns
                  <<",\"infer_start_ns\":"<<p.frame.infer_start_ns<<",\"infer_end_ns\":"<<p.frame.infer_end_ns
                  <<",\"pair_delta_us\":"<<p.frame.pair_delta_us<<",\"pair_status\":\"software_raw_timestamp_match\""
                  <<",\"dropped_capacity\":"<<p.frame.dropped_capacity<<",\"dropped_age\":"<<p.frame.dropped_age
                  <<",\"dropped_mismatch\":"<<p.frame.dropped_mismatch
                  <<",\"dropped_superseded\":"<<p.frame.dropped_superseded
                  <<",\"source_size\":["<<source_w<<","<<source_h<<"],\"roi_xywh\":["<<rx<<","<<ry<<","<<rw<<","<<rh<<"]"
                  <<",\"backend\":\""<<mode<<"\",\"pl_calls\":"<<profiles[i].calls;
            if(i>=2)result<<",\"source_gap_us\":"<<p.frame.timestamp_us-camera_profiles[i-2].frame.timestamp_us
                <<",\"skipped_sequence_frames\":"<<uint32_t(p.frame.sequence-camera_profiles[i-2].frame.sequence)-1;
            else result<<",\"source_gap_us\":0,\"skipped_sequence_frames\":0";
#endif
            result<<",\"keypoints\":[";
            for(int c=0;c<17;++c)result<<(c?",":"")<<"["<<p.points[c*3]<<","<<p.points[c*3+1]<<","<<p.points[c*3+2]<<"]";
            result<<"]}";
        }
        result<<"]";
#ifdef U02_STEREO_PIPELINE
#ifdef U02_STEREO_PAIR_FIRST
        result<<",\"stereo_preparation\":\"pair_first_parallel_decode_roi\",\"decode_profile_scope\":\"selected_pair_overlapped_with_previous_right_inference\"";
#else
        result<<",\"stereo_preparation\":\"parallel_capture_workers_exact_decode_roi\",\"decode_profile_scope\":\"worker_computation_overlapped_with_other_camera_and_inference\"";
#endif
#ifdef U02_IMAGE_ROI_DECODE
        result<<",\"jpeg_region_decode\":true";
#endif
#endif
        std::ofstream snapshot(std::string(argv[5])+".jpg",std::ios::binary);
        snapshot.write(reinterpret_cast<const char*>(encoded.data()),encoded.size());
        snapshot.close();if(!snapshot)throw std::runtime_error("Camera snapshot write failed");
#endif
        std::ofstream tensor(std::string(argv[5])+".input.f32",std::ios::binary);
        for(int c=0;c<3;++c)tensor.write((const char*)(const float*)input.channel(c),96*128*4);
        tensor.close();if(!tensor)throw std::runtime_error("Pipeline input write failed");
#endif
        result<<"}\n";result.close();if(!result) throw std::runtime_error("Output write failed");
        std::ofstream heat(std::string(argv[5])+".heatmap",std::ios::binary);
        for(int c=0;c<17;++c) heat.write((const char*)(const float*)out.channel(c),768*4);
        heat.close();if(!heat) throw std::runtime_error("Heatmap write failed");
        std::cout<<"LIGHT_POSE_OK backend="<<mode<<" infer_ms="<<ms<<" valid_joints="<<valid<<"\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
