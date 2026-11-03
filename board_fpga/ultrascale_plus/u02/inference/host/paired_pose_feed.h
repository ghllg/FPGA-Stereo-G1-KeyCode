#pragma once
#include "pose_feed.h"
#include "camera_capture.h"
#include "stereo_capture.h"
#include <fstream>

class PairedPoseFeed {
    PoseFeed transport;
    CameraFrame left;
    std::vector<float> left_points;
    uint64_t left_calls=0;
    std::array<double,4> left_roi{};
    uint64_t expired_pair_drops=0;
    std::array<uint64_t,2> previous_timestamp{{0,0}};
    std::array<uint32_t,2> previous_sequence{{0,0}};
    std::string boot;
#ifdef U02_STREAM_PIPELINE
    unsigned inflight_pairs;
#endif
    void camera(std::ostringstream& json,const CameraFrame& frame,const std::vector<float>& points,
                uint64_t calls,int width,int height,double rx,double ry,double rw,double rh) {
        if(points.size()!=51 || calls!=110)throw std::runtime_error("Expected complete FPGA COCO17 result");
        json<<"{\"camera_id\":\""<<(frame.camera_id?"right":"left")<<"\",\"frame_id\":"<<frame.sequence
            <<",\"capture_timestamp_us\":"<<frame.timestamp_us<<",\"timestamp_source\":\"v4l2_monotonic_unverified_exposure\""
            <<",\"capture_flags\":"<<frame.flags<<",\"receive_monotonic_ns\":"<<frame.receive_monotonic_ns
            <<",\"dequeue_monotonic_ns\":"<<frame.dequeue_monotonic_ns<<",\"infer_start_ns\":"<<frame.infer_start_ns
            <<",\"infer_end_ns\":"<<frame.infer_end_ns<<",\"forward_ms\":"<<(frame.infer_end_ns-frame.infer_start_ns)/1e6
            <<",\"source_size\":["<<width<<","<<height<<"],\"roi_xywh\":["<<rx<<","<<ry<<","<<rw<<","<<rh<<"]"
            <<",\"source_gap_us\":"<<(previous_timestamp[frame.camera_id]?frame.timestamp_us-previous_timestamp[frame.camera_id]:0)
            <<",\"skipped_sequence_frames\":"<<(previous_timestamp[frame.camera_id]?uint32_t(frame.sequence-previous_sequence[frame.camera_id])-1:0)
            <<",\"coordinate_space\":\"source_image_pixels\",\"pl_calls\":"<<calls<<",\"keypoints\":[";
        for(unsigned i=0;i<17;++i) {
            float x=points[i*3],y=points[i*3+1],score=points[i*3+2];
            if(!std::isfinite(x+y+score))throw std::runtime_error("Nonfinite COCO17 message");
            bool valid=score>=.3f && x>=0 && x<width && y>=0 && y<height;
            json<<(i?",":"")<<"["<<x<<","<<y<<","<<score<<","<<(valid?"true":"false")<<"]";
        }
        json<<"]}";
        previous_timestamp[frame.camera_id]=frame.timestamp_us;previous_sequence[frame.camera_id]=frame.sequence;
    }
public:
#ifdef U02_STREAM_PIPELINE
    explicit PairedPoseFeed(unsigned pairs):transport("192.168.50.1",18084,true),inflight_pairs(pairs) {
        if(pairs<1 || pairs>2)throw std::runtime_error("Invalid pipeline pair capacity");
#else
    PairedPoseFeed():transport("192.168.50.1",18084,true) {
#endif
        std::ifstream file("/proc/sys/kernel/random/boot_id");file>>boot;
        if(boot.size()!=36 || boot.find_first_not_of("0123456789abcdef-")!=std::string::npos)
            throw std::runtime_error("Cannot identify board boot for paired timestamps");
    }
    bool wait_for_client(unsigned timeout_ms) {
        if(timeout_ms>10000)throw std::runtime_error("Bounded client wait required");
        const auto deadline=stereo_monotonic_ns()+uint64_t(timeout_ms)*1000000ull;
        do {
            if(transport.accept_client())return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }while(stereo_monotonic_ns()<deadline);
        return false;
    }
    void submit(const CameraFrame& frame,const std::vector<float>& points,uint64_t calls,
                int width,int height,double rx,double ry,double rw,double rh) {
        transport.flush();
        if(!frame.camera_id){left=frame;left_points=points;left_calls=calls;left_roi={{rx,ry,rw,rh}};return;}
        if(left_points.size()!=51 || left.pair_id!=frame.pair_id)
            throw std::runtime_error("Paired result identity mismatch");
        const auto probe=transport.receive_probe();
        const auto published=stereo_monotonic_ns();
        std::ostringstream json;json<<std::setprecision(9)<<"{\"schema\":2,\"message_type\":\"paired_coco17\""
            <<",\"session_id\":"<<transport.session_id()<<",\"boot_id\":\""<<boot<<"\",\"pair_id\":"<<frame.pair_id
#ifdef U02_STREAM_DW_CHANNEL_LOCAL
            <<",\"boundary_quantization\":\"dw_channel_local\""
#endif
            <<",\"whole_network_on_board\":true,\"backend\":\"pl-stream\""
#if defined(U02_STREAM_PIPELINE)
            <<",\"stream_candidate\":\"shared_model_int32_pipeline\",\"stream_device_abi\":9,\"stream_packet_version\":1"
            <<",\"stream_contexts\":"<<inflight_pairs*2<<",\"inflight_pairs\":"<<inflight_pairs<<",\"pl_cores\":1"
#elif defined(U02_STEREO_DUAL_CONTEXT)
            <<",\"stream_candidate\":\"dual_context_int32_resident\",\"stream_device_abi\":7,\"stream_packet_version\":1"
            <<",\"stream_contexts\":2,\"pl_cores\":1"
#elif defined(U02_STREAM_FP32_OUTPUT)
            <<",\"stream_candidate\":\"fp32_direct\""
#elif defined(U02_FORMAL_STEREO_ABI3) && defined(U02_STREAM_RESIDENT_PACKETS)
            <<",\"stream_candidate\":\"ingress_int32_resident\",\"stream_device_abi\":3,\"stream_packet_version\":1"
#else
#error PairedPoseFeed requires an explicitly identified FPGA streaming backend
#endif
            <<",\"keypoint_order\":\"COCO17\",\"valid_confidence_threshold\":0.3,\"exposure_sync_verified\":false"
            <<",\"pair_delta_us\":"<<frame.pair_delta_us<<",\"pair_status\":\""<<u02_stereo_pair_policy::status<<"\""
#if defined(U02_STEREO_HALF_FRAME_RAW) || defined(U02_STEREO_NEAREST_RAW_20MS)
            <<",\"pairing_policy\":\""<<u02_stereo_pair_policy::name<<"\",\"pair_limit_us\":"<<u02_stereo_pair_policy::limit_us
#endif
            <<",\"publish_monotonic_ns\":"<<published<<",\"clock_probe\":{\"nonce\":"<<probe.nonce
            <<",\"pc_send_ns\":"<<probe.pc_ns<<",\"board_receive_ns\":"<<probe.board_receive_ns
            <<",\"board_send_ns\":"<<published<<"},\"network_dropped_pairs\":"<<transport.dropped
            <<",\"network_tcp_srtt_us\":"<<transport.tcp_rtt_us()
            <<",\"network_tcp_rtt_scope\":\"kernel_smoothed_ack_round_trip_not_one_way_or_inference\""
            <<",\"expired_pair_drops\":"<<expired_pair_drops
            <<",\"dropped_capacity\":"<<frame.dropped_capacity<<",\"dropped_age\":"<<frame.dropped_age
            <<",\"dropped_mismatch\":"<<frame.dropped_mismatch<<",\"dropped_superseded\":"<<frame.dropped_superseded<<",\"cameras\":[";
        camera(json,left,left_points,left_calls,width,height,left_roi[0],left_roi[1],left_roi[2],left_roi[3]);json<<",";
        camera(json,frame,points,calls,width,height,rx,ry,rw,rh);json<<"]}";
        transport.send_metadata(json.str());left_points.clear();
    }
    uint64_t sent() const{return transport.sent;}
    uint64_t dropped() const{return transport.dropped;}
    void flush(){transport.flush();}
    void discard_expired_pair(){left_points.clear();++expired_pair_drops;}
    uint64_t expired() const{return expired_pair_drops;}
};
