// Experimental ABI9 live capture integration; formal ABI7 entry is unchanged.
#include "pose_network_worker.h"
#include "backend_stream_pipeline.h"
#include "stereo_capture.h"
#include "paired_pose_feed.h"
#include "async_paired_pose_feed.h"
#include "pose_diagnostic_snapshots.h"
#include <csignal>
#include <deque>
#include <iostream>
#include <cstdlib>
#include <sys/resource.h>

static volatile std::sig_atomic_t stop_requested=0;
static void request_stop(int) {stop_requested=1;}
static int env_int(const char* key,int fallback,int low,int high) {
    const char* value=std::getenv(key);size_t used=0;int result=value?std::stoi(value,&used):fallback;
    if((value && value[used]) || result<low || result>high)throw std::runtime_error(std::string("Invalid ")+key);
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
    unsigned slot;
    uint64_t pair_submit_ns,pair_complete_ns;
};
static void write_frame(std::ostream& out,const CameraHistory& h,int width,int height,double rx,double ry,double rw,double rh) {
    const auto& f=h.frame;
    out<<"{\"camera_id\":\""<<(f.camera_id?"right":"left")<<"\",\"pair_id\":"<<f.pair_id
       <<",\"frame_id\":"<<f.sequence<<",\"sequence\":"<<f.sequence<<",\"timestamp_us\":"<<f.timestamp_us
       <<",\"flags\":"<<f.flags<<",\"timestamp_source\":\"v4l2_monotonic_unverified_exposure\""
       <<",\"receive_monotonic_ns\":"<<f.receive_monotonic_ns<<",\"dequeue_monotonic_ns\":"<<f.dequeue_monotonic_ns
       <<",\"infer_start_ns\":"<<f.infer_start_ns<<",\"infer_end_ns\":"<<f.infer_end_ns
       <<",\"pair_delta_us\":"<<f.pair_delta_us<<",\"pair_status\":\""<<u02_stereo_pair_policy::status<<"\""
       <<",\"pipeline_slot\":"<<h.slot<<",\"pair_submit_ns\":"<<h.pair_submit_ns
       <<",\"pair_complete_ns\":"<<h.pair_complete_ns
       <<",\"pl_calls\":"<<h.timing.calls<<",\"source_size\":["<<width<<","<<height
       <<"],\"roi_xywh\":["<<rx<<","<<ry<<","<<rw<<","<<rh<<"],\"keypoints\":[";
    const auto precision=out.precision();out<<std::setprecision(9);
    for(int i=0;i<17;++i)out<<(i?",":"")<<"["<<h.points[3*i]<<","<<h.points[3*i+1]<<","<<h.points[3*i+2]<<"]";
    out.precision(precision);
    out<<"]}";
}
int main(int argc,char** argv) {
    try {
        if(argc!=12)throw std::runtime_error("Usage: pose_stereo_pipeline cpu|pl-stream PARAM BIN LEFT_NODE,RIGHT_NODE OUTPUT.json SOURCE_W SOURCE_H ROI_X ROI_Y ROI_W ROI_H");
        const std::string mode=argv[1],execution=std::getenv("POSE_PAIR_EXECUTION")?std::getenv("POSE_PAIR_EXECUTION"):"parallel";
        if(mode!="cpu" && mode!="pl-stream")throw std::runtime_error("Live stereo requires a complete CPU or explicit ABI9 backend");
        if(execution!="serial" && execution!="parallel")throw std::runtime_error("Invalid stereo execution");
        const unsigned pairs=env_int("POSE_INFLIGHT_PAIRS",2,1,2);
        const int threads=env_int("POSE_THREADS",1,1,1),seconds=env_int("POSE_RUN_SECONDS",120,2,3600);
        const int network=env_int("POSE_PAIR_NETWORK",mode=="pl-stream"?1:0,0,1);
        const bool hybrid=env_int("POSE_HYBRID_JPEG",0,0,1);
        if(hybrid && !network)throw std::runtime_error("Hybrid mode requires paired FPGA transport");
        PoseDiagnosticSnapshots snapshots(env_int("POSE_DEBUG_SNAPSHOTS",0,0,1));
        if(mode=="cpu" && network)throw std::runtime_error("CPU camera baseline is local-only; cannot label CPU as FPGA on wire");
        const int width=std::stoi(argv[6]),height=std::stoi(argv[7]);
        const double rx=std::stod(argv[8]),ry=std::stod(argv[9]),rw=std::stod(argv[10]),rh=std::stod(argv[11]);
        if(width!=u02_stereo_source::width || height!=u02_stereo_source::height || !std::isfinite(rx+ry+rw+rh) || rw<=0 || rh<=0)
#ifdef U02_STEREO_SOURCE_1080P
            throw std::runtime_error("Invalid 1080p stereo geometry");
#else
            throw std::runtime_error("Invalid 720p stereo geometry");
#endif
        const auto rois=pose_view_rois(width,height,rx,ry,rw,rh);
        std::signal(SIGINT,request_stop);std::signal(SIGTERM,request_stop);
        std::shared_ptr<StreamPipelineDevice> device;
        if(mode=="pl-stream")device=std::make_shared<StreamPipelineDevice>(pairs*2);
        struct Slot {
            ncnn::Mat inputs[2];
            CameraFrame frames[2];
            PoseImageTimes preparations[2];
            std::vector<uint8_t> encoded[2];
            std::unique_ptr<PoseNetworkSession> sessions[2];
            // Destruction joins workers before destroying sessions or inputs.
            std::unique_ptr<PoseWorker> workers[2];
            uint64_t submitted=0;
            bool active=false;
        };
        std::array<Slot,2> slots;
        for(unsigned slot=0;slot<pairs;++slot)for(unsigned side=0;side<2;++side) {
            auto& s=slots[slot];s.inputs[side]=ncnn::Mat(96,128,3);
            if(s.inputs[side].empty() || s.inputs[side].cstep!=96*128)
                throw std::runtime_error("Private contiguous camera input required");
            std::unique_ptr<Backend> backend;
            if(device)backend.reset(new StreamPipelineBackend(device,slot*2+side));
            s.sessions[side].reset(new PoseNetworkSession(mode,argv[2],argv[3],threads,std::move(backend)));
        }
        if(device)device->seal();
        for(unsigned slot=0;slot<pairs;++slot)for(unsigned side=0;side<2;++side)
            slots[slot].workers[side].reset(new PoseWorker(*slots[slot].sessions[side],slots[slot].inputs[side]));
        std::unique_ptr<PairedPoseFeed> feed;std::unique_ptr<AsyncPairedPoseFeed> raw_feed;
        if(network){if(hybrid)raw_feed.reset(new AsyncPairedPoseFeed(pairs));else feed.reset(new PairedPoseFeed(pairs));}
        const int wait_client_ms=env_int("POSE_PAIR_WAIT_CLIENT_MS",0,0,10000);
        if(wait_client_ms && !feed && !raw_feed)throw std::runtime_error("Client gate requires the actual FPGA network sender");
        const auto network_wait_start=stereo_monotonic_ns();
        if(wait_client_ms && !(raw_feed?raw_feed->wait_for_client(wait_client_ms):feed->wait_for_client(wait_client_ms)))
            throw std::runtime_error("PC did not connect before the bounded capture start gate");
        const auto network_wait_ns=stereo_monotonic_ns()-network_wait_start;
        StereoCapture camera(argv[4],width,height,rx,ry,rw,rh);
        const auto a=stereo_monotonic_ns(),b=PoseNetworkSession::now_ns(),c=stereo_monotonic_ns();
        if(b<a || b>c)throw std::runtime_error("Worker/capture monotonic clocks differ");
        uint64_t completed_pairs=0,total_pl_calls=0,expired=0;
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
        uint64_t total_ioctl_cpu_ns=0,total_forward_cpu_ns=0,total_ioctl_wall_ns=0,total_driver_wait_ns=0,total_profile_calls=0,total_profile_polls=0;
#endif
        ncnn::Mat last_inputs[2]={ncnn::Mat(96,128,3),ncnn::Mat(96,128,3)};
        if(last_inputs[0].empty() || last_inputs[1].empty() || last_inputs[0].cstep!=96*128 || last_inputs[1].cstep!=96*128)
            throw std::runtime_error("Private final input snapshots required");
        std::vector<uint8_t> last_encoded[2];
        std::deque<CameraHistory> history;
        std::deque<unsigned> pending;
        unsigned last_slot=0;
        rusage cpu_before{},cpu_after{};
        if(getrusage(RUSAGE_SELF,&cpu_before))throw std::runtime_error("Process CPU clock unavailable");
        const auto run_start=stereo_monotonic_ns();auto run_end=run_start;
        auto capturing=[&] {return !stop_requested && stereo_monotonic_ns()-run_start<uint64_t(seconds)*1000000000ull;};
        auto submit=[&](unsigned index) {
            auto& s=slots[index];
            if(s.active)throw std::runtime_error("Cannot overwrite an in-flight camera pair");
            for(;;) {
                if(!capturing())return false;
                try {
                    // Copy both fresh tensors into this slot's private storage.
                    // Capture's current_input may be replaced by the next view.
                    for(unsigned side=0;side<2;++side) {
                        s.frames[side]=camera.next(s.encoded[side]);
                        if(s.frames[side].camera_id!=side)throw std::runtime_error("Unexpected stereo view order");
                        s.preparations[side]=camera.prepared_input(static_cast<float*>(s.inputs[side].data));
                    }
                    break;
                }catch(const StereoPairExpired&) {
                    camera.discard_expired_pending();++expired;if(feed)feed->discard_expired_pair();if(raw_feed)raw_feed->discard_expired_pair();
                }
            }
            if(s.frames[0].pair_id!=s.frames[1].pair_id)throw std::runtime_error("Prepared pair identity differs");
            s.submitted=stereo_monotonic_ns();s.active=true;
            s.workers[0]->submit();if(execution=="serial")s.workers[0]->wait();
            s.workers[1]->submit();pending.push_back(index);return true;
        };
        for(unsigned index=0;index<pairs;++index)if(!submit(index))break;
        while(!pending.empty()) {
            const unsigned index=pending.front();pending.pop_front();auto& s=slots[index];
            s.workers[0]->wait();s.workers[1]->wait();
            const auto completed=stereo_monotonic_ns();
            std::vector<float> points[2];
            for(unsigned side=0;side<2;++side) {
                const auto& session=*s.sessions[side];const auto& t=session.timing;
                if(t.calls!=(device?110u:0u))throw std::runtime_error("Incomplete network PL call count");
                if(device && (session.stem->calls!=1 || session.stem->macs!=2654208))
                    throw std::runtime_error("Expected one complete known ARM integer stem");
                s.frames[side].infer_start_ns=t.start_ns;s.frames[side].infer_end_ns=t.end_ns;
                const auto& r=rois[side];
                points[side]=session.points(r[0],r[1],r[2],r[3]);total_pl_calls+=t.calls;
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
                if(t.ioctl_cpu.rows.size()!=t.calls)throw std::runtime_error("Incomplete ioctl CPU samples");
                total_ioctl_cpu_ns+=t.ioctl_cpu.cpu_ns;total_forward_cpu_ns+=t.forward_thread_cpu_ns;
                total_profile_calls+=t.ioctl_cpu.rows.size();
                for(const auto& sample:t.ioctl_cpu.rows) {
                    total_ioctl_wall_ns+=sample.wall_ns;total_driver_wait_ns+=sample.driver_ns;total_profile_polls+=sample.polls;
                }
#endif
            }
            // Single publisher drains in capture order, regardless of worker
            // completion order. Stop submits no new pairs and drains both slots.
            if(feed)for(unsigned side=0;side<2;++side)
                feed->submit(s.frames[side],points[side],s.sessions[side]->timing.calls,width,height,
                             rois[side][0],rois[side][1],rois[side][2],rois[side][3]);
            if(raw_feed){const uint64_t calls[2]={s.sessions[0]->timing.calls,s.sessions[1]->timing.calls};
                raw_feed->submit(s.frames,points,s.encoded,calls,width,height,rois);}
            if(snapshots.due(completed)) {
                PoseDiagnosticSnapshots::View views[2];
                for(unsigned side=0;side<2;++side)
                    views[side]={s.frames[side],s.inputs[side],s.sessions[side]->output,s.encoded[side],points[side]};
                snapshots.add(completed,views[0],views[1]);
            }
            for(unsigned side=0;side<2;++side) {
                // A final capture attempt can expire after copying only the
                // left view. Preserve the last completed pair independently.
                memcpy(last_inputs[side].data,s.inputs[side].data,96*128*3*4);
                last_encoded[side]=std::move(s.encoded[side]);
            }
            for(unsigned side=0;side<2;++side)
                history.push_back({s.frames[side],std::move(points[side]),s.sessions[side]->timing,s.preparations[side],index,s.submitted,completed});
            while(history.size()>128) {history.pop_front();history.pop_front();}
            ++completed_pairs;run_end=stereo_monotonic_ns();last_slot=index;s.active=false;
            // Only this completed slot can receive another fresh camera pair.
            if(capturing())submit(index);
        }
        if(getrusage(RUSAGE_SELF,&cpu_after))throw std::runtime_error("Process CPU clock unavailable");
        const auto cpu_observed_end=stereo_monotonic_ns();
        camera.stop();if(feed)feed->flush();if(raw_feed)raw_feed->finish();
        if(!completed_pairs || history.size()%2)throw std::runtime_error("No complete stereo result");
        const std::string output=argv[5];
        snapshots.save(output,rois);
        const auto& last=slots[last_slot];
        for(unsigned side=0;side<2;++side) {
            const std::string prefix=output+(side?".right.json":".left.json");
            const auto& r=rois[side];
            last.sessions[side]->save(prefix,width,height,r[0],r[1],r[2],r[3]);
            save_input(prefix,last_inputs[side],last_encoded[side]);
        }
        std::ofstream result(output);if(!result)throw std::runtime_error("Cannot save stereo result");
        const double elapsed=(run_end-run_start)*1e-9;
        result<<std::setprecision(12)<<"{\"scope\":\"real_stereo_capture_decode_two_complete_board_networks\",\"backend\":\""<<mode
            <<"\",\"stream_candidate\":\""<<(device?"shared_model_int32_pipeline":"arm_only_private_sessions")<<"\",\"stream_device_abi\":"<<(device?9:0)
            <<",\"diagnostic_snapshot_pairs\":"<<snapshots.pairs.size()
            <<",\"inflight_pairs\":"<<pairs<<",\"stream_contexts\":"<<(device?pairs*2:0)
            <<",\"pl_cores\":"<<(device?1:0)<<",\"ordered_publish\":true,\"drain_on_stop\":true"
            <<",\"pairing_policy\":\""<<u02_stereo_pair_policy::name<<"\",\"pair_limit_us\":"<<u02_stereo_pair_policy::limit_us
            <<",\"stream_packet_version\":"<<(device?1:0)<<",\"whole_network_on_board\":true,\"execution\":\""<<execution
            <<"\",\"stem_variant\":\""<<(device?"int16_smlal_int32_accumulate":"ncnn_fp32_cpu")
            <<"\",\"cpu_threads\":"<<threads<<",\"requested_run_seconds\":"<<seconds
            <<",\"completed_pairs\":"<<completed_pairs<<",\"total_image_forwards\":"<<completed_pairs*2
            <<",\"total_pl_calls\":"<<total_pl_calls<<",\"camera_elapsed_s\":"<<elapsed
            <<",\"observed_completed_pairs_per_s\":"<<completed_pairs/elapsed
            <<",\"process_cpu_user_ms\":"<<((cpu_after.ru_utime.tv_sec-cpu_before.ru_utime.tv_sec)*1000.0+(cpu_after.ru_utime.tv_usec-cpu_before.ru_utime.tv_usec)*.001)
            <<",\"process_cpu_system_ms\":"<<((cpu_after.ru_stime.tv_sec-cpu_before.ru_stime.tv_sec)*1000.0+(cpu_after.ru_stime.tv_usec-cpu_before.ru_stime.tv_usec)*.001)
            <<",\"process_cpu_window_ms\":"<<(cpu_observed_end-run_start)*1e-6
            <<",\"process_cpu_scope\":\"all_process_threads_including_decode_not_kernel_irq_or_other_processes\""
#ifdef U02_IMAGE_WORKSPACE
            <<",\"image_preprocess_variant\":\"exact_sparse_rows_workspace_v2\",\"image_workspace_reuse_level\":"<<U02_IMAGE_WORKSPACE_REUSE_LEVEL
#endif
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
            <<",\"ioctl_thread_cpu_profile\":true,\"profiling_overhead_included\":true,\"cpu_clock\":\"CLOCK_THREAD_CPUTIME_ID\""
            <<",\"ioctl_cpu_scope\":\"calling_thread_user_plus_kernel_inside_actual_ioctl_excludes_submission_queue_and_other_threads\""
            <<",\"all_forward_thread_cpu_ns\":"<<total_forward_cpu_ns<<",\"all_ioctl_thread_cpu_ns\":"<<total_ioctl_cpu_ns
            <<",\"all_ioctl_wall_ns\":"<<total_ioctl_wall_ns<<",\"all_driver_wait_ns\":"<<total_driver_wait_ns
            <<",\"all_profile_calls\":"<<total_profile_calls<<",\"all_profile_polls\":"<<total_profile_polls
#endif
            <<",\"expired_pair_drops\":"<<expired<<",\"network_enabled\":"<<(network?"true":"false")
            <<",\"network_start_gate_ms\":"<<wait_client_ms<<",\"network_start_wait_ms\":"<<network_wait_ns*1e-6
            <<",\"paired_network_frames_sent\":"<<(raw_feed?raw_feed->sent():(feed?feed->sent():0))<<",\"paired_network_frames_dropped\":"<<(raw_feed?raw_feed->dropped():(feed?feed->dropped():0))
            <<",\"hybrid_jpeg_enabled\":"<<(hybrid?"true":"false")<<",\"sender_superseded_pairs\":"<<(raw_feed?raw_feed->superseded():0)
            <<",\"history_limit_images\":128,\"history_scope\":\"last_complete_pairs\",\"exposure_sync_verified\":false"
            <<",\"camera_requested_fps\":30,\"coordinate_space\":\"source_image_pixels\",\"keypoint_order\":\"COCO17\""
            <<",\"timing_scope\":\"overlapping_per_image_wall_times_not_additive\",\"camera_frames\":[";
        for(size_t i=0;i<history.size();++i) {
            if(i)result<<",";const auto& r=rois[history[i].frame.camera_id];
            write_frame(result,history[i],width,height,r[0],r[1],r[2],r[3]);
        }
        result<<"],\"forward_profiles\":[";
        for(size_t i=0;i<history.size();++i) {
            if(i)result<<",";const auto& h=history[i];const auto& t=h.timing;
            result<<"{\"forward_ms\":"<<(t.end_ns-t.start_ns)*1e-6<<",\"prepare_ms\":"<<t.prepare_ms<<",\"pl_wait_ms\":"<<t.wait_ms
                <<",\"submission_queue_ms\":"<<t.queue_ms<<",\"control_ms\":"<<t.control_ms
                <<",\"quantize_ms\":"<<t.quantize_ms<<",\"dequantize_ms\":"<<t.dequantize_ms
                <<",\"decode_ms\":"<<h.preparation.decode_ms<<",\"preprocess_ms\":"<<h.preparation.preprocess_ms;
#ifdef U02_IOCTL_THREAD_CPU_PROFILE
            result<<",\"forward_thread_cpu_ns\":"<<t.forward_thread_cpu_ns
                  <<",\"ioctl_thread_cpu_ns\":"<<t.ioctl_cpu.cpu_ns<<",\"ioctl_cpu_samples\":[";
            for(size_t call=0;call<t.ioctl_cpu.rows.size();++call) {
                const auto& s=t.ioctl_cpu.rows[call];if(call)result<<",";
                result<<"{\"call_index\":"<<call<<",\"thread_cpu_ns\":"<<s.cpu_ns<<",\"ioctl_wall_ns\":"<<s.wall_ns
                      <<",\"driver_wait_ns\":"<<s.driver_ns<<",\"polls\":"<<s.polls
                      <<",\"a_bytes\":"<<s.a_bytes<<",\"b_bytes\":"<<s.b_bytes<<",\"c_bytes\":"<<s.c_bytes<<",\"ops\":"<<s.ops<<"}";
            }
            result<<"]";
#endif
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
        std::cout<<"PIPELINE_STEREO_COMPLETE pairs="<<completed_pairs<<" total_pl_calls="<<total_pl_calls<<" pairs_per_s="<<completed_pairs/elapsed<<"\n";
        return 0;
    }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
