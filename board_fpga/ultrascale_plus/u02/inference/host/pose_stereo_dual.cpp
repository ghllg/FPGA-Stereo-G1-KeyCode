#include "pose_network_worker.h"
#include "stereo_capture.h"
#include "paired_pose_feed.h"
#include <csignal>
#include <deque>
#include <iostream>
#include <cstdlib>

static volatile std::sig_atomic_t stop_requested=0;
static void request_stop(int) {stop_requested=1;}
static int env_int(const char* key,int fallback,int low,int high) {
    const char* value=std::getenv(key);int result=value?std::stoi(value):fallback;
    if(result<low || result>high)throw std::runtime_error(std::string("Invalid ")+key);
    return result;
}
static void save_input(const std::string& prefix,const ncnn::Mat& input,const std::vector<uint8_t>& jpeg) {
    std::ofstream tensor(prefix+".input.f32",std::ios::binary),image(prefix+".jpg",std::ios::binary);
    for(int c=0;c<3;++c)tensor.write(reinterpret_cast<const char*>(static_cast<const float*>(input.channel(c))),96*128*4);
    image.write(reinterpret_cast<const char*>(jpeg.data()),jpeg.size());
    if(!tensor || !image)throw std::runtime_error("Cannot save real camera reference inputs");
}
struct CameraHistory {
    CameraFrame frame;
    std::vector<float> points;
    PoseNetworkSession::Timing timing;
    PoseImageTimes preparation;
};
static void write_frame(std::ostream& out,const CameraHistory& h,int width,int height,double rx,double ry,double rw,double rh) {
    const auto& f=h.frame;
    out<<"{\"camera_id\":\""<<(f.camera_id?"right":"left")<<"\",\"pair_id\":"<<f.pair_id
       <<",\"frame_id\":"<<f.sequence<<",\"sequence\":"<<f.sequence<<",\"timestamp_us\":"<<f.timestamp_us
       <<",\"flags\":"<<f.flags<<",\"timestamp_source\":\"v4l2_monotonic_unverified_exposure\""
       <<",\"receive_monotonic_ns\":"<<f.receive_monotonic_ns<<",\"dequeue_monotonic_ns\":"<<f.dequeue_monotonic_ns
       <<",\"infer_start_ns\":"<<f.infer_start_ns<<",\"infer_end_ns\":"<<f.infer_end_ns
       <<",\"pair_delta_us\":"<<f.pair_delta_us<<",\"pair_status\":\"software_raw_timestamp_match\""
       <<",\"pl_calls\":"<<h.timing.calls<<",\"source_size\":["<<width<<","<<height
       <<"],\"roi_xywh\":["<<rx<<","<<ry<<","<<rw<<","<<rh<<"],\"keypoints\":[";
    const auto precision=out.precision();out<<std::setprecision(9);
    for(int i=0;i<17;++i)out<<(i?",":"")<<"["<<h.points[3*i]<<","<<h.points[3*i+1]<<","<<h.points[3*i+2]<<"]";
    out.precision(precision);
    out<<"]}";
}
int main(int argc,char** argv) {
    try {
        if(argc!=12)throw std::runtime_error("Usage: pose_stereo_dual cpu|pl-stream PARAM BIN LEFT_NODE,RIGHT_NODE OUTPUT.json SOURCE_W SOURCE_H ROI_X ROI_Y ROI_W ROI_H");
        const std::string mode=argv[1],execution=std::getenv("POSE_PAIR_EXECUTION")?std::getenv("POSE_PAIR_EXECUTION"):"parallel";
        if(mode!="cpu" && mode!="pl-stream")throw std::runtime_error("Live stereo requires a complete CPU or explicit ABI7 backend");
        if(execution!="serial" && execution!="parallel")throw std::runtime_error("Invalid stereo execution");
        const int threads=env_int("POSE_THREADS",2,1,4),seconds=env_int("POSE_RUN_SECONDS",120,2,3600);
        const int network=env_int("POSE_PAIR_NETWORK",mode=="pl-stream"?1:0,0,1);
        if(mode=="cpu" && network)throw std::runtime_error("CPU camera baseline is local-only; cannot label CPU as FPGA on wire");
        const int width=std::stoi(argv[6]),height=std::stoi(argv[7]);
        const double rx=std::stod(argv[8]),ry=std::stod(argv[9]),rw=std::stod(argv[10]),rh=std::stod(argv[11]);
        if(width!=u02_stereo_source::width || height!=u02_stereo_source::height || !std::isfinite(rx+ry+rw+rh) || rw<=0 || rh<=0)
#ifdef U02_STEREO_SOURCE_1080P
            throw std::runtime_error("Invalid 1080p stereo geometry");
#else
            throw std::runtime_error("Invalid 720p stereo geometry");
#endif
        std::signal(SIGINT,request_stop);std::signal(SIGTERM,request_stop);
        ncnn::Mat inputs[2]={ncnn::Mat(96,128,3),ncnn::Mat(96,128,3)};
        ncnn::Mat last_inputs[2]={ncnn::Mat(96,128,3),ncnn::Mat(96,128,3)};
        if(inputs[0].cstep!=96*128 || inputs[1].cstep!=96*128 ||
           last_inputs[0].cstep!=96*128 || last_inputs[1].cstep!=96*128)
            throw std::runtime_error("Contiguous stereo input/snapshot allocation required");
        std::shared_ptr<StreamDualDevice> device;
        std::unique_ptr<Backend> backends[2];
        if(mode=="pl-stream") {
            device=std::make_shared<StreamDualDevice>();
            for(unsigned side=0;side<2;++side)backends[side].reset(new StreamDualBackend(device,side));
        }
        PoseNetworkSession left(mode,argv[2],argv[3],threads,std::move(backends[0]));
        PoseNetworkSession right(mode,argv[2],argv[3],threads,std::move(backends[1]));
        PoseNetworkSession* sessions[2]={&left,&right};
        PoseWorker left_worker(left,inputs[0]),right_worker(right,inputs[1]);
        std::unique_ptr<PairedPoseFeed> feed;if(network)feed.reset(new PairedPoseFeed);
        const int wait_client_ms=env_int("POSE_PAIR_WAIT_CLIENT_MS",0,0,10000);
        if(wait_client_ms && !feed)throw std::runtime_error("Client gate requires the actual FPGA network sender");
        const auto network_wait_start=stereo_monotonic_ns();
        if(feed && wait_client_ms && !feed->wait_for_client(wait_client_ms))
            throw std::runtime_error("PC did not connect before the bounded capture start gate");
        const auto network_wait_ns=stereo_monotonic_ns()-network_wait_start;
        StereoCapture camera(argv[4],width,height,rx,ry,rw,rh);
        const auto a=stereo_monotonic_ns(),b=PoseNetworkSession::now_ns(),c=stereo_monotonic_ns();
        if(b<a || b>c)throw std::runtime_error("Worker/capture monotonic clocks differ");
        uint64_t completed_pairs=0,total_pl_calls=0,expired=0;
        std::vector<uint8_t> encoded[2],last_encoded[2];std::deque<CameraHistory> history;
        const auto run_start=stereo_monotonic_ns();auto run_end=run_start;
        while(!stop_requested && stereo_monotonic_ns()-run_start<uint64_t(seconds)*1000000000ull) {
            CameraFrame frames[2];PoseImageTimes preparations[2];
            try {
                // Copy each normalized tensor before next() replaces capture's
                // current_input. No inference begins until both are owned.
                for(unsigned side=0;side<2;++side) {
                    frames[side]=camera.next(encoded[side]);
                    if(frames[side].camera_id!=side)throw std::runtime_error("Unexpected stereo view order");
                    preparations[side]=camera.prepared_input(static_cast<float*>(inputs[side].data));
                }
            }catch(const StereoPairExpired&) {
                camera.discard_expired_pending();++expired;if(feed)feed->discard_expired_pair();continue;
            }
            if(frames[0].pair_id!=frames[1].pair_id)throw std::runtime_error("Prepared pair identity differs");
            left_worker.submit();if(execution=="serial")left_worker.wait();
            right_worker.submit();left_worker.wait();right_worker.wait();
            std::vector<float> points[2];
            for(unsigned side=0;side<2;++side) {
                const auto& t=sessions[side]->timing;
                if(t.calls!=(mode=="pl-stream"?110u:0u))throw std::runtime_error("Incomplete network PL call count");
                if(mode=="pl-stream" && (sessions[side]->stem->calls!=1 || sessions[side]->stem->macs!=2654208))
                    throw std::runtime_error("Expected one complete known ARM integer stem");
                frames[side].infer_start_ns=t.start_ns;frames[side].infer_end_ns=t.end_ns;
                points[side]=sessions[side]->points(rx,ry,rw,rh);total_pl_calls+=t.calls;
            }
            // Publish only after both complete results exist. Timestamp order
            // need not be left-before-right in this overlapping execution.
            if(feed)for(unsigned side=0;side<2;++side)feed->submit(frames[side],points[side],sessions[side]->timing.calls,width,height,rx,ry,rw,rh);
            for(unsigned side=0;side<2;++side) {
                memcpy(last_inputs[side].data,inputs[side].data,96*128*3*4);
                last_encoded[side]=std::move(encoded[side]);
            }
            for(unsigned side=0;side<2;++side)history.push_back({frames[side],std::move(points[side]),sessions[side]->timing,preparations[side]});
            while(history.size()>128) {history.pop_front();history.pop_front();}
            ++completed_pairs;run_end=stereo_monotonic_ns();
        }
        camera.stop();if(feed)feed->flush();
        if(!completed_pairs || history.size()%2)throw std::runtime_error("No complete stereo result");
        const std::string output=argv[5];
        for(unsigned side=0;side<2;++side) {
            const std::string prefix=output+(side?".right.json":".left.json");
            sessions[side]->save(prefix,width,height,rx,ry,rw,rh);save_input(prefix,last_inputs[side],last_encoded[side]);
        }
        std::ofstream result(output);if(!result)throw std::runtime_error("Cannot save stereo result");
        const double elapsed=(run_end-run_start)*1e-9;
        result<<std::setprecision(12)<<"{\"scope\":\"real_stereo_capture_decode_two_complete_board_networks\",\"backend\":\""<<mode
            <<"\",\"stream_candidate\":\""<<(device?"dual_context_int32_resident":"arm_only_two_session")<<"\",\"stream_device_abi\":"<<(device?7:0)
            <<",\"stream_packet_version\":"<<(device?1:0)<<",\"whole_network_on_board\":true,\"execution\":\""<<execution
            <<"\",\"stem_variant\":\""<<(device?"int16_smlal_int32_accumulate":"ncnn_fp32_cpu")
            <<"\",\"cpu_threads\":"<<threads<<",\"requested_run_seconds\":"<<seconds
            <<",\"completed_pairs\":"<<completed_pairs<<",\"total_image_forwards\":"<<completed_pairs*2
            <<",\"total_pl_calls\":"<<total_pl_calls<<",\"camera_elapsed_s\":"<<elapsed
            <<",\"observed_completed_pairs_per_s\":"<<completed_pairs/elapsed
            <<",\"expired_pair_drops\":"<<expired<<",\"network_enabled\":"<<(feed?"true":"false")
            <<",\"network_start_gate_ms\":"<<wait_client_ms<<",\"network_start_wait_ms\":"<<network_wait_ns*1e-6
            <<",\"paired_network_frames_sent\":"<<(feed?feed->sent():0)<<",\"paired_network_frames_dropped\":"<<(feed?feed->dropped():0)
            <<",\"history_limit_images\":128,\"history_scope\":\"last_complete_pairs\",\"exposure_sync_verified\":false"
            <<",\"camera_requested_fps\":30,\"coordinate_space\":\"source_image_pixels\",\"keypoint_order\":\"COCO17\""
            <<",\"timing_scope\":\"overlapping_per_image_wall_times_not_additive\",\"camera_frames\":[";
        for(size_t i=0;i<history.size();++i) {if(i)result<<",";write_frame(result,history[i],width,height,rx,ry,rw,rh);}
        result<<"],\"forward_profiles\":[";
        for(size_t i=0;i<history.size();++i) {
            if(i)result<<",";const auto& h=history[i];const auto& t=h.timing;
            result<<"{\"forward_ms\":"<<(t.end_ns-t.start_ns)*1e-6<<",\"prepare_ms\":"<<t.prepare_ms<<",\"pl_wait_ms\":"<<t.wait_ms
                <<",\"submission_queue_ms\":"<<t.queue_ms<<",\"control_ms\":"<<t.control_ms
                <<",\"quantize_ms\":"<<t.quantize_ms<<",\"dequantize_ms\":"<<t.dequantize_ms
                <<",\"decode_ms\":"<<h.preparation.decode_ms<<",\"preprocess_ms\":"<<h.preparation.preprocess_ms;
#ifdef U02_STREAM_BOUNDARY_PROFILE
            result<<",\"boundary_range_ms\":"<<t.boundary_range_ms<<",\"boundary_quant_ms\":"<<t.boundary_quant_ms
                  <<",\"boundary_coeff_ms\":"<<t.boundary_coeff_ms<<",\"boundary_patch_ms\":"<<t.boundary_patch_ms;
#endif
#ifdef U02_STREAM_RANGE_CACHE
            result<<",\"range_hit_elements\":"<<t.range_hit_elements
                  <<",\"range_verified_elements\":"<<t.range_verified_elements
                  <<",\"range_published_elements\":"<<t.range_published_elements;
#endif
            result<<"}";
        }
        result<<"]";camera.write_timing_diagnostic(result);result<<"}\n";
        if(!result)throw std::runtime_error("Stereo result write failed");
        std::cout<<"DUAL_STEREO_COMPLETE pairs="<<completed_pairs<<" total_pl_calls="<<total_pl_calls<<" pairs_per_s="<<completed_pairs/elapsed<<"\n";
        return 0;
    }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
