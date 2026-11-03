#pragma once
#include "camera_capture.h"
#include "image_preprocess.h"
#include "stereo_source_geometry.h"
#include "stereo_pair_policy.h"
#ifdef U02_STEREO_NEAREST_RAW_20MS
#include "stereo_nearest_pair.h"
#endif
#include "pose_view_roi.h"
#if defined(U02_1080P_STARTUP_QUALIFIED) && (!defined(U02_STEREO_SOURCE_1080P) || !defined(U02_STEREO_COORDINATED_START))
#error The isolated 1080p startup qualifier requires explicit 1080p and coordinated start
#endif
#include <algorithm>
#if defined(U02_STEREO_PREFETCH_THREE) && (!defined(U02_STEREO_PAIR_FIRST) || !defined(U02_FORMAL_STEREO_ABI3))
#error Three-pair prefetch requires the verified pair-first formal capture path
#endif
#include <cstring>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <sstream>
#if defined(U02_STEREO_STARTUP_QUALIFIED) || defined(U02_1080P_STARTUP_QUALIFIED)
#include <iostream>
#endif
#include <time.h>
#if defined(U02_STEREO_STAGE_TRACE) && (!defined(U02_STEREO_PAIR_FIRST) || !defined(U02_FORMAL_STEREO_ABI3) || defined(U02_STEREO_PREFETCH_THREE))
#error Stage timing requires the original single prepared pair path
#endif

inline uint64_t stereo_monotonic_ns() {
    timespec time{};
    if(clock_gettime(CLOCK_MONOTONIC,&time))throw std::runtime_error("Monotonic clock unavailable");
    return uint64_t(time.tv_sec)*1000000000ull+time.tv_nsec;
}
class StereoPairExpired : public std::runtime_error {
public: explicit StereoPairExpired(const std::string& message):std::runtime_error(message){}
};

// Two independent producers, at most four owned JPEGs per source. The sole
// inference caller consumes an actual pair left then right; PL stays serialized.
class StereoCapture {
    struct Owned {CameraFrame frame;std::vector<uint8_t> jpeg;std::vector<float> input;PoseImageTimes preparation;
#ifdef U02_STEREO_STAGE_TRACE
        std::array<uint64_t,7> stages{{0,0,0,0,0,0,0}};
#endif
    };
    std::array<std::unique_ptr<CameraCapture>,2> cameras;
    std::array<std::deque<Owned>,2> queues;
    std::array<std::thread,2> workers;
#ifdef U02_IMAGE_WORKSPACE
    std::array<PoseImageWorkspace,2> image_workspaces;
#endif
#ifdef U02_STEREO_PAIR_FIRST
    std::thread pair_worker;
#ifdef U02_STEREO_PREFETCH_THREE
    std::deque<std::array<Owned,2>> prepared_pairs;
    size_t prepared_high_watermark=0;
#else
    std::array<Owned,2> prepared_pair;
    bool pair_ready=false;
#endif
#endif
    std::mutex mutex;
    std::condition_variable ready;
    std::atomic<bool> stopping{false};
    std::exception_ptr failure;
    Owned pending;
    std::vector<float> current_input;
    PoseImageTimes current_preparation;
    std::array<std::array<double,4>,2> roi;
    bool has_pending=false;
    uint64_t pair_count=0,capacity_drops=0,age_drops=0,mismatch_drops=0,superseded_drops=0;
#ifdef U02_STEREO_STAGE_TRACE
    std::deque<std::array<uint64_t,7>> stage_trace;
    uint64_t stage_trace_count=0;
#endif
#ifdef U02_STEREO_COORDINATED_START
    std::array<uint64_t,2> streamon_requested_ns{{0,0}},streamon_completed_ns{{0,0}};
    void start_configured_cameras(unsigned right_delay_us=0) {
        std::mutex launch_mutex;
        std::condition_variable launch_ready;
        bool launch=false;
        std::array<std::exception_ptr,2> errors;
        std::array<std::thread,2> starters;
        auto release=[&] {
            {std::lock_guard<std::mutex> lock(launch_mutex);launch=true;}
            launch_ready.notify_all();
            for(auto& worker:starters)if(worker.joinable())worker.join();
        };
        try {
            for(unsigned side=0;side<2;++side)starters[side]=std::thread([&,side] {
                {std::unique_lock<std::mutex> lock(launch_mutex);launch_ready.wait(lock,[&]{return launch;});}
                try {
                    if(side==1 && right_delay_us)std::this_thread::sleep_for(std::chrono::microseconds(right_delay_us));
                    streamon_requested_ns[side]=stereo_monotonic_ns();
                    cameras[side]->start();
                    streamon_completed_ns[side]=stereo_monotonic_ns();
                }catch(...){errors[side]=std::current_exception();}
            });
        }catch(...){release();throw;}
        release();
        for(const auto& error:errors)if(error)std::rethrow_exception(error);
    }
#endif
#if defined(U02_STEREO_STARTUP_QUALIFIED) || defined(U02_1080P_STARTUP_QUALIFIED)
    std::vector<std::string> startup_attempts;
    uint64_t startup_duration_ns=0;
    void start_qualified_cameras(const std::string& nodes,int width,int height,bool exercise_retry) {
        if((nodes!="/dev/video0,/dev/video2" && nodes!="/dev/video2,/dev/video0") || width!=u02_stereo_source::width || height!=u02_stereo_source::height)
            throw std::runtime_error("Qualified stereo requires resolved capture nodes and configured source geometry");
        const auto started=stereo_monotonic_ns();
        const size_t comma=nodes.find(',');
        // A bounded software phase search, not an exposure trigger. Every
        // attempt uses unmodified kernel timestamps and the declared startup gate.
        for(unsigned attempt=0;attempt<3;++attempt) {
            if(attempt==0)cameras[0].reset(new CameraCapture(nodes.substr(0,comma),width,height,false));
            cameras[1].reset(new CameraCapture(nodes.substr(comma+1),width,height,false));
            if(cameras[0]->bus_id()!="usb-xhci-hcd.0.auto-1.4" || cameras[1]->bus_id()!="usb-xhci-hcd.0.auto-1.2")
                throw std::runtime_error("Qualified camera physical identity changed");
            const unsigned delay_us=attempt*8000;
            uint64_t retry_anchor_us=0;
            if(attempt==0)start_configured_cameras();
            else {
                // Preserve the left stream's phase. Anchor the retry request
                // to a newly dequeued left frame rather than restarting both.
                std::vector<uint8_t> jpeg;
                retry_anchor_us=cameras[0]->next(jpeg).timestamp_us;
                std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
                streamon_requested_ns[1]=stereo_monotonic_ns();
                cameras[1]->start();streamon_completed_ns[1]=stereo_monotonic_ns();
            }
            std::array<std::vector<CameraFrame>,2> samples;
            std::array<std::exception_ptr,2> errors;
            auto collect=[&](unsigned side) {
                try {
                    std::vector<uint8_t> jpeg;uint64_t previous=0;
                    for(unsigned index=0;index<24;++index) {
                        auto frame=cameras[side]->next(jpeg);
                        frame.receive_monotonic_ns=stereo_monotonic_ns();
                        if((frame.flags&0xe000)!=0x2000 || frame.timestamp_us<=previous)
                            throw std::runtime_error("Startup timestamps must increase and be monotonic");
                        previous=frame.timestamp_us;
                        if(index>=6)samples[side].push_back(frame);
                    }
                }catch(...){errors[side]=std::current_exception();}
            };
            std::thread left([&]{collect(0);});collect(1);left.join();
            for(const auto& error:errors)if(error)std::rethrow_exception(error);
            unsigned matches=0;
#ifdef U02_STEREO_NEAREST_RAW_20MS
            const auto timestamp=[](const CameraFrame& f){return f.timestamp_us;};
            for(const auto& left:samples[0])for(const auto& right:samples[1])
                if(u02_stereo_nearest::mutual(left.timestamp_us,right.timestamp_us,
                    samples[0],samples[1],timestamp,u02_stereo_pair_policy::startup_limit_us))++matches;
#else
            size_t l=0,r=0;
            while(l<samples[0].size() && r<samples[1].size()) {
                const auto delta=int64_t(samples[1][r].timestamp_us)-int64_t(samples[0][l].timestamp_us);
                if(delta>u02_stereo_pair_policy::startup_limit_us)++l;
                else if(delta< -u02_stereo_pair_policy::startup_limit_us)++r;
                else {++matches;++l;++r;}
            }
#endif
            const bool qualified=matches>=12;
            // Camera-only diagnostic can exercise the retry even if its first
            // real window is valid. It is reported separately, never as a
            // spontaneous startup failure or a reliability success trial.
            const bool forced_retry=exercise_retry && attempt==0;
            const bool accepted=qualified && !forced_retry;
            std::ostringstream diagnostic;
            diagnostic<<"{\"attempt\":"<<attempt+1<<",\"right_request_delay_us\":"<<delay_us
                <<",\"restart_scope\":\""<<(attempt==0?"both":"right_only")<<"\",\"retry_anchor_left_timestamp_us\":"<<retry_anchor_us
                <<",\"sample_frames_per_side\":18,\"matches\":"<<matches
                <<",\"required_matches\":12,\"pair_limit_us\":"<<u02_stereo_pair_policy::startup_limit_us<<",\"qualified\":"<<(qualified?"true":"false")
                <<",\"diagnostic_forced_retry\":"<<(forced_retry?"true":"false")<<",\"accepted\":"<<(accepted?"true":"false")
                <<",\"streamon_requested_ns\":["<<streamon_requested_ns[0]<<","<<streamon_requested_ns[1]<<"]"
                <<",\"streamon_completed_ns\":["<<streamon_completed_ns[0]<<","<<streamon_completed_ns[1]<<"]"
                <<",\"raw_timestamp_us\":[";
            for(unsigned side=0;side<2;++side) {
                if(side)diagnostic<<",";diagnostic<<"[";
                for(size_t i=0;i<samples[side].size();++i){if(i)diagnostic<<",";diagnostic<<samples[side][i].timestamp_us;}
                diagnostic<<"]";
            }
            diagnostic<<"]}";startup_attempts.push_back(diagnostic.str());
            std::cerr<<"U02_STARTUP_PAIR_QUALIFICATION="<<diagnostic.str()<<std::endl;
            if(accepted){startup_duration_ns=stereo_monotonic_ns()-started;return;}
            cameras[1]->stop();cameras[1].reset();
        }
        throw std::runtime_error("U02_STARTUP_PAIR_QUALIFICATION_FAILED after three bounded camera-only attempts");
    }
#endif
#ifdef U02_FORMAL_STEREO_ABI3
    std::array<std::deque<std::array<uint64_t,3>>,2> raw_timing;
    std::array<uint64_t,2> raw_counts{{0,0}};
#ifdef U02_STEREO_RAW_TRACE
    std::array<std::deque<std::array<uint64_t,4>>,2> capture_details;
    std::array<uint64_t,2> drained_ready_counts{{0,0}};
#endif
    // Caller holds mutex; timestamps stay raw. Nearest20 also consults this history.
    void write_raw_timing(std::ostream& out
#ifdef U02_STEREO_RAW_TRACE
        ,bool include_trace=true
#endif
    ) {
        out<<"{\"pair_limit_us\":"<<u02_stereo_pair_policy::limit_us
#ifdef U02_STEREO_COORDINATED_START
           <<",\"startup_mode\":\"configured_then_parallel_streamon_requests\""
           <<",\"streamon_requested_ns\":["<<streamon_requested_ns[0]<<","<<streamon_requested_ns[1]<<"]"
           <<",\"streamon_completed_ns\":["<<streamon_completed_ns[0]<<","<<streamon_completed_ns[1]<<"]"
#endif
#if defined(U02_STEREO_STARTUP_QUALIFIED) || defined(U02_1080P_STARTUP_QUALIFIED)
           <<",\"startup_qualification_duration_ns\":"<<startup_duration_ns<<",\"startup_attempts\":[";
        for(size_t i=0;i<startup_attempts.size();++i){if(i)out<<",";out<<startup_attempts[i];}
        out<<"]"
#endif
           <<",\"raw_counts\":["<<raw_counts[0]<<","<<raw_counts[1]
           <<"],\"pairs_prepared\":"<<pair_count<<",\"mismatch_drops\":"<<mismatch_drops
#ifdef U02_STEREO_PREFETCH_THREE
           <<",\"prepared_queue_capacity\":3,\"prepared_queue_depth\":"<<prepared_pairs.size()
           <<",\"prepared_queue_high_watermark\":"<<prepared_high_watermark
#endif
#ifdef U02_STEREO_RAW_TRACE
           <<",\"raw_trace_limit_per_camera\":12000,\"raw_trace_complete\":"
           <<((include_trace && raw_timing[0].size()==raw_counts[0] && raw_timing[1].size()==raw_counts[1])?"true":"false")
           <<",\"raw_trace_scope\":\"frames_returned_by_CameraCapture_next_not_all_V4L2_dequeues_or_verified_exposures\""
#endif
           <<",\"entry_fields\":[\"sequence\",\"timestamp_us\",\"receive_monotonic_ns\"],\"recent_raw\":[";
        for(unsigned side=0;side<2;++side) {
            if(side)out<<",";out<<"[";
            size_t first=0;
#ifdef U02_STEREO_RAW_TRACE
            if(!include_trace && raw_timing[side].size()>16)first=raw_timing[side].size()-16;
#endif
            for(size_t i=first;i<raw_timing[side].size();++i) {
                const auto& p=raw_timing[side][i];if(i!=first)out<<",";
                out<<"["<<p[0]<<","<<p[1]<<","<<p[2]<<"]";
            }
            out<<"]";
        }
        out<<"]";
#ifdef U02_STEREO_RAW_TRACE
        out<<",\"raw_returned_frames_only\":true,\"total_drained_ready_frames\":["
           <<drained_ready_counts[0]<<","<<drained_ready_counts[1]<<"]"
           <<",\"capture_details_complete\":"
           <<((include_trace && capture_details[0].size()==raw_counts[0] && capture_details[1].size()==raw_counts[1])?"true":"false")
           <<",\"capture_details_fields\":[\"sequence\",\"discarded_ready_frames\",\"capture_call_ns\",\"producer_lock_acquired_ns\"]"
           <<",\"recent_capture_details\":[";
        for(unsigned side=0;side<2;++side) {
            if(side)out<<",";out<<"[";
            const size_t first=(!include_trace && capture_details[side].size()>16)?capture_details[side].size()-16:0;
            for(size_t i=first;i<capture_details[side].size();++i) {
                const auto& p=capture_details[side][i];if(i!=first)out<<",";
                out<<"["<<p[0]<<","<<p[1]<<","<<p[2]<<","<<p[3]<<"]";
            }
            out<<"]";
        }
        out<<"]";
#endif
        out<<"}";
    }
    std::string pair_wait_error() {
        std::ostringstream out;out<<"No prepared stereo pair within bounded wait U02_PAIRING_RAW=";
        write_raw_timing(out
#ifdef U02_STEREO_RAW_TRACE
            ,false
#endif
        );return out.str();
    }
#endif
    static bool expired(const Owned& frame,uint64_t now) {
        return now<frame.frame.receive_monotonic_ns || now-frame.frame.receive_monotonic_ns>500000000ull ||
               now/1000<frame.frame.timestamp_us || now/1000-frame.frame.timestamp_us>500000ull;
    }
    void produce(unsigned side) {
        try {
            uint64_t previous=0;
            while(!stopping) {
                Owned item;item.frame=cameras[side]->next(item.jpeg);
                item.frame.camera_id=side;item.frame.receive_monotonic_ns=stereo_monotonic_ns();
                if((item.frame.flags&0xe000)!=0x2000 || item.frame.timestamp_us<=previous)
                    throw std::runtime_error("Stereo requires strictly increasing V4L2 monotonic timestamps");
                previous=item.frame.timestamp_us;
#ifndef U02_STEREO_PAIR_FIRST
                item.input.resize(96*128*3);
#ifdef U02_IMAGE_ROI_DECODE
                item.preparation=prepare_pose_image_roi(item.jpeg,u02_stereo_source::width,u02_stereo_source::height,roi[side][0],roi[side][1],roi[side][2],roi[side][3],item.input.data());
#else
                item.preparation=prepare_pose_image(item.jpeg,u02_stereo_source::width,u02_stereo_source::height,roi[side][0],roi[side][1],roi[side][2],roi[side][3],item.input.data());
#endif
#endif
                std::lock_guard<std::mutex> guard(mutex);
#ifdef U02_STEREO_RAW_TRACE
                const uint64_t producer_lock_ns=stereo_monotonic_ns();
#endif
#ifdef U02_FORMAL_STEREO_ABI3
                ++raw_counts[side];raw_timing[side].push_back({{item.frame.sequence,item.frame.timestamp_us,item.frame.receive_monotonic_ns}});
#ifdef U02_STEREO_RAW_TRACE
                drained_ready_counts[side]+=item.frame.discarded_ready_frames;
                capture_details[side].push_back({{item.frame.sequence,item.frame.discarded_ready_frames,
                    uint64_t(item.frame.capture_ms*1000000.),producer_lock_ns}});
                if(capture_details[side].size()>12000)capture_details[side].pop_front();
                if(raw_timing[side].size()>12000)raw_timing[side].pop_front();
#else
                if(raw_timing[side].size()>16)raw_timing[side].pop_front();
#endif
#endif
                if(expired(item,stereo_monotonic_ns())){++age_drops;continue;}
                if(queues[side].size()==4){queues[side].pop_front();++capacity_drops;}
                queues[side].push_back(std::move(item));ready.notify_all();
            }
        } catch(...) {
            std::lock_guard<std::mutex> guard(mutex);
            if(!stopping)failure=std::current_exception();
            stopping=true;ready.notify_all();
        }
    }
#ifdef U02_STEREO_PAIR_FIRST
    void prepare_pairs() {
        try {
            while(!stopping) {
                std::array<Owned,2> pair;
#ifdef U02_STEREO_STAGE_TRACE
                pair[0].stages[1]=stereo_monotonic_ns();
#endif
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    ready.wait(lock,[&]{return stopping ||
#ifdef U02_STEREO_PREFETCH_THREE
                        prepared_pairs.size()<3;
#else
                        (!pair_ready && !has_pending);
#endif
                    });
                    if(stopping)break;
#ifdef U02_STEREO_STAGE_TRACE
                    const uint64_t slot_begin=pair[0].stages[1];
                    const uint64_t match_begin=stereo_monotonic_ns();
#endif
                    for(;;) {
                        if(stopping)return;
                        const auto now=stereo_monotonic_ns();
                        for(auto& queue:queues)while(!queue.empty() && expired(queue.front(),now)){queue.pop_front();++age_drops;}
                        size_t best_l=0,best_r=0;uint64_t latest=0;
#ifdef U02_STEREO_NEAREST_RAW_20MS
                        latest=u02_stereo_nearest::latest(queues[0],queues[1],raw_timing[0],raw_timing[1],
                            [](const Owned& item){return item.frame.timestamp_us;},
                            [](const std::array<uint64_t,3>& item){return item[1];},
                            u02_stereo_pair_policy::limit_us,best_l,best_r);
#else
                        for(size_t l=0;l<queues[0].size();++l)for(size_t r=0;r<queues[1].size();++r) {
                            auto lt=queues[0][l].frame.timestamp_us,rt=queues[1][r].frame.timestamp_us;
                            auto delta=int64_t(rt)-int64_t(lt);
                            if(delta>=-u02_stereo_pair_policy::limit_us && delta<=u02_stereo_pair_policy::limit_us && lt+rt>latest){best_l=l;best_r=r;latest=lt+rt;}
                        }
#endif
                        if(latest) {
                            for(size_t i=0;i<best_l;++i){queues[0].pop_front();++superseded_drops;}
                            for(size_t i=0;i<best_r;++i){queues[1].pop_front();++superseded_drops;}
                            for(unsigned side=0;side<2;++side){pair[side]=std::move(queues[side].front());queues[side].pop_front();}
#ifdef U02_STEREO_STAGE_TRACE
                            pair[0].stages[0]=pair_count;
                            pair[0].stages[1]=slot_begin;
                            pair[0].stages[2]=match_begin;
                            pair[0].stages[3]=stereo_monotonic_ns();
#endif
                            const auto delta=int64_t(pair[1].frame.timestamp_us)-int64_t(pair[0].frame.timestamp_us);
                            for(auto& image:pair){image.frame.pair_id=pair_count;image.frame.pair_delta_us=delta;}
                            ++pair_count;break;
                        }
#ifndef U02_STEREO_NEAREST_RAW_20MS
                        if(!queues[0].empty() && !queues[1].empty()) {
                            auto delta=int64_t(queues[1].front().frame.timestamp_us)-int64_t(queues[0].front().frame.timestamp_us);
                            queues[delta>0?0:1].pop_front();++mismatch_drops;continue;
                        }
#endif
                        // Nearest20 waits for lookahead instead of prematurely discarding a neighbour.
                        ready.wait(lock);
                    }
                }
                // Decode only this real pair, while the prior right forward runs.
                std::array<std::exception_ptr,2> errors;
                auto decode=[&](unsigned side) {
                    try {
                        pair[side].input.resize(96*128*3);
                        pair[side].preparation=prepare_pose_image_roi(pair[side].jpeg,u02_stereo_source::width,u02_stereo_source::height,
                            roi[side][0],roi[side][1],roi[side][2],roi[side][3],pair[side].input.data()
#ifdef U02_IMAGE_WORKSPACE
                            ,image_workspaces[side]
#endif
                        );
                    }catch(...){errors[side]=std::current_exception();}
                };
                std::thread left([&]{decode(0);});decode(1);left.join();
                for(auto& error:errors)if(error)std::rethrow_exception(error);
#ifdef U02_STEREO_STAGE_TRACE
                pair[0].stages[4]=stereo_monotonic_ns();
#endif
                std::lock_guard<std::mutex> lock(mutex);
                if(stopping)break;
#ifdef U02_STEREO_STAGE_TRACE
                pair[0].stages[5]=stereo_monotonic_ns();
#endif
                // A selected pair stays identifiable even if processing delayed it.
                for(auto& image:pair){image.frame.dropped_capacity=capacity_drops;image.frame.dropped_age=age_drops;
                    image.frame.dropped_mismatch=mismatch_drops;image.frame.dropped_superseded=superseded_drops;}
#ifdef U02_STEREO_PREFETCH_THREE
                // Only this producer pushes. Consumption while decode runs
                // can only free slots; each queued pair owns its JPEG/tensor.
                if(prepared_pairs.size()>=3)throw std::runtime_error("Prepared pair capacity exceeded");
                prepared_pairs.push_back(std::move(pair));
                prepared_high_watermark=std::max(prepared_high_watermark,prepared_pairs.size());
#else
                prepared_pair=std::move(pair);pair_ready=true;
#endif
                ready.notify_all();
            }
        }catch(...) {
            std::lock_guard<std::mutex> lock(mutex);failure=std::current_exception();stopping=true;ready.notify_all();
        }
    }
#endif
public:
#ifdef U02_FORMAL_STEREO_ABI3
    void write_timing_diagnostic(std::ostream& out) {
        std::lock_guard<std::mutex> guard(mutex);out<<",\"pairing_diagnostic\":";write_raw_timing(out);
#ifdef U02_STEREO_STAGE_TRACE
        out<<",\"pair_stage_trace\":{\"limit\":6000,\"observed_left_inputs\":"<<stage_trace_count
           <<",\"complete\":"<<(stage_trace.size()==stage_trace_count?"true":"false")
           <<",\"fields\":[\"pair_id\",\"slot_wait_begin_ns\",\"matching_begin_ns\",\"selected_ns\",\"decode_end_ns\",\"published_ns\",\"consumed_ns\"],\"rows\":[";
        for(size_t i=0;i<stage_trace.size();++i) {
            if(i)out<<",";out<<"[";
            for(unsigned j=0;j<7;++j){if(j)out<<",";out<<stage_trace[i][j];}
            out<<"]";
        }
        out<<"]}";
#endif
    }
#endif
    StereoCapture(const std::string& nodes,int width,int height,double rx,double ry,double rw,double rh,bool exercise_retry=false):roi(pose_view_rois(width,height,rx,ry,rw,rh)) {
#if defined(U02_STEREO_STARTUP_QUALIFIED) || defined(U02_1080P_STARTUP_QUALIFIED)
        start_qualified_cameras(nodes,width,height,exercise_retry);
#else
#ifdef U02_FORMAL_STEREO_ABI3
        // Linux video numbers can reverse after boot/rebinding. The caller
        // resolves the fixed left/right USB ports; QUERYCAP below still checks
        // the exact physical identities before starting producer workers.
        if((nodes!="/dev/video0,/dev/video2" && nodes!="/dev/video2,/dev/video0") || width!=u02_stereo_source::width || height!=u02_stereo_source::height)
#ifdef U02_STEREO_SOURCE_1080P
            throw std::runtime_error("1080p candidate requires two resolved capture nodes and 1920x1080 MJPEG");
#else
            throw std::runtime_error("Formal stereo requires two resolved capture nodes and 1280x720 MJPEG");
#endif
        const size_t separator=nodes.find(',');
#ifdef U02_STEREO_COORDINATED_START
        constexpr bool start_immediately=false;
#else
        constexpr bool start_immediately=true;
#endif
        cameras[0].reset(new CameraCapture(nodes.substr(0,separator),width,height,start_immediately));
        cameras[1].reset(new CameraCapture(nodes.substr(separator+1),width,height,start_immediately));
#else
        if(nodes!="/dev/video0,/dev/video2" || width!=1280 || height!=720)
            throw std::runtime_error("Stereo requires known capture nodes and 1280x720 MJPEG");
        cameras[0].reset(new CameraCapture("/dev/video0",width,height));
        cameras[1].reset(new CameraCapture("/dev/video2",width,height));
#endif
        if(cameras[0]->bus_id()!="usb-xhci-hcd.0.auto-1.4" || cameras[1]->bus_id()!="usb-xhci-hcd.0.auto-1.2")
            throw std::runtime_error("Stereo camera physical identity changed");
#ifdef U02_STEREO_COORDINATED_START
        start_configured_cameras();
#endif
#endif
        try {for(unsigned i=0;i<2;++i)workers[i]=std::thread([this,i]{produce(i);});
#ifdef U02_STEREO_PAIR_FIRST
            pair_worker=std::thread([this]{prepare_pairs();});
#endif
        }
        catch(...){stop();throw;}
    }
    ~StereoCapture(){stop();}
    PoseImageTimes prepared_input(float* destination) {
        if(!destination || current_input.size()!=96*128*3)throw std::runtime_error("Stereo prepared input missing");
        std::memcpy(destination,current_input.data(),current_input.size()*sizeof(float));
        return current_preparation;
    }
    void stop() {
        stopping=true;ready.notify_all();
        for(auto& worker:workers)if(worker.joinable())worker.join();
#ifdef U02_STEREO_PAIR_FIRST
        if(pair_worker.joinable())pair_worker.join();
#endif
        for(auto& camera:cameras)if(camera)camera->stop();
    }
    CameraFrame next(std::vector<uint8_t>& jpeg) {
        std::unique_lock<std::mutex> lock(mutex);
        if(failure)std::rethrow_exception(failure);
        if(has_pending) {
            const auto now=stereo_monotonic_ns();
            if(expired(pending,now))throw StereoPairExpired("Paired right image expired during left inference source_age_us="+
                std::to_string(int64_t(now/1000)-int64_t(pending.frame.timestamp_us))+" receive_age_ns="+
                std::to_string(int64_t(now)-int64_t(pending.frame.receive_monotonic_ns)));
            pending.frame.dequeue_monotonic_ns=now;
            has_pending=false;jpeg=std::move(pending.jpeg);current_input=std::move(pending.input);
            ready.notify_all();
            current_preparation=pending.preparation;return pending.frame;
        }
#ifdef U02_STEREO_PAIR_FIRST
        const auto timeout=std::chrono::steady_clock::now()+std::chrono::seconds(3);
#ifdef U02_STEREO_PREFETCH_THREE
        for(;;) {
            if(!ready.wait_until(lock,timeout,[&]{return stopping || !prepared_pairs.empty();}))
                throw std::runtime_error(pair_wait_error());
            if(failure)std::rethrow_exception(failure);
            if(stopping)throw std::runtime_error("Stereo capture stopped");
            const auto now=stereo_monotonic_ns();
            if(expired(prepared_pairs.front()[0],now) || expired(prepared_pairs.front()[1],now)) {
                // Drop an entire expired pair before lending either side.
                prepared_pairs.pop_front();age_drops+=2;ready.notify_all();continue;
            }
            auto pair=std::move(prepared_pairs.front());prepared_pairs.pop_front();
            Owned left=std::move(pair[0]);pending=std::move(pair[1]);has_pending=true;
            left.frame.dequeue_monotonic_ns=now;jpeg=std::move(left.jpeg);
            current_input=std::move(left.input);current_preparation=left.preparation;
            ready.notify_all();return left.frame;
        }
#else
        if(!ready.wait_until(lock,timeout,[&]{return stopping || pair_ready;}))throw std::runtime_error(
#ifdef U02_FORMAL_STEREO_ABI3
            pair_wait_error()
#else
            "No prepared stereo pair within bounded wait"
#endif
            );
        if(failure)std::rethrow_exception(failure);
        if(!pair_ready)throw std::runtime_error("Stereo capture stopped");
        Owned left=std::move(prepared_pair[0]);pending=std::move(prepared_pair[1]);pair_ready=false;has_pending=true;
        left.frame.dequeue_monotonic_ns=stereo_monotonic_ns();jpeg=std::move(left.jpeg);
#ifdef U02_STEREO_STAGE_TRACE
        left.stages[6]=left.frame.dequeue_monotonic_ns;
        stage_trace.push_back(left.stages);++stage_trace_count;
        if(stage_trace.size()>6000)stage_trace.pop_front();
#endif
        current_input=std::move(left.input);current_preparation=left.preparation;ready.notify_all();return left.frame;
#endif
#else
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
        for(;;) {
            if(failure)std::rethrow_exception(failure);
            const uint64_t now=stereo_monotonic_ns();
            for(auto& queue:queues)while(!queue.empty() && expired(queue.front(),now)){queue.pop_front();++age_drops;}
            if(!queues[0].empty() && !queues[1].empty()) {
                // Select the newest real match, so slower inference does not
                // consume the oldest of four buffered frames repeatedly.
                size_t best_left=0,best_right=0;uint64_t latest=0;
                for(size_t l=0;l<queues[0].size();++l)for(size_t r=0;r<queues[1].size();++r) {
                    const auto lt=queues[0][l].frame.timestamp_us,rt=queues[1][r].frame.timestamp_us;
                    const auto delta=int64_t(rt)-int64_t(lt);
                    if(delta>=-u02_stereo_pair_policy::limit_us && delta<=u02_stereo_pair_policy::limit_us && lt+rt>latest) {
                        best_left=l;best_right=r;latest=lt+rt;
                    }
                }
                if(latest) {
                    while(best_left--){queues[0].pop_front();++superseded_drops;}
                    while(best_right--){queues[1].pop_front();++superseded_drops;}
                    // Reserve half the age budget for the serialized pair.
                    if(now/1000-queues[0].front().frame.timestamp_us>250000 ||
                       now/1000-queues[1].front().frame.timestamp_us>250000) {
                        queues[0].pop_front();queues[1].pop_front();age_drops+=2;continue;
                    }
                }
                const int64_t delta=int64_t(queues[1].front().frame.timestamp_us)-int64_t(queues[0].front().frame.timestamp_us);
                if(delta>u02_stereo_pair_policy::limit_us){queues[0].pop_front();++mismatch_drops;continue;}
                if(delta< -u02_stereo_pair_policy::limit_us){queues[1].pop_front();++mismatch_drops;continue;}
                Owned left=std::move(queues[0].front());queues[0].pop_front();
                pending=std::move(queues[1].front());queues[1].pop_front();has_pending=true;
                for(auto* item:{&left,&pending}) {
                    item->frame.pair_id=pair_count;item->frame.pair_delta_us=delta;
                    item->frame.dropped_capacity=capacity_drops;item->frame.dropped_age=age_drops;
                    item->frame.dropped_mismatch=mismatch_drops;
                    item->frame.dropped_superseded=superseded_drops;
                }
                left.frame.dequeue_monotonic_ns=stereo_monotonic_ns();
                ++pair_count;jpeg=std::move(left.jpeg);current_input=std::move(left.input);
                current_preparation=left.preparation;return left.frame;
            }
            if(stopping)throw std::runtime_error("Stereo capture stopped");
            if(ready.wait_until(lock,deadline)==std::cv_status::timeout)throw std::runtime_error("No stereo pair within bounded wait");
        }
#endif
    }
    bool at_pair_boundary() {
        std::lock_guard<std::mutex> lock(mutex);return !has_pending;
    }
    void discard_expired_pending() {
        std::lock_guard<std::mutex> lock(mutex);
        if(!has_pending || !expired(pending,stereo_monotonic_ns()))
            throw std::runtime_error("Cannot discard a nonexpired stereo half-pair");
        pending=Owned();has_pending=false;++age_drops;ready.notify_all();
    }
};
