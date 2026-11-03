#pragma once
#include "paired_pose_feed.h"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

// Inference owns the capture slots. This sender owns copies of complete pairs:
// one waiting latest pair plus PoseFeed's one bounded pending TCP body.
class AsyncPairedPoseFeed {
    struct Pair {
        CameraFrame frame[2];
        std::vector<float> points[2];
        std::vector<uint8_t> jpeg[2];
        uint64_t calls[2];
        std::array<std::array<double,4>,2> rois;
        int width,height;
        uint64_t fixture_publish_ns=0;
    };
    PairedPoseFeed feed;
    const bool fixture_mode;
    std::mutex mutex;
    std::condition_variable ready;
    std::unique_ptr<Pair> queued;
    std::thread worker;
    std::exception_ptr error;
    std::atomic<uint64_t> replaced{0},expired_pairs{0};
    bool stopping=false;
    uint64_t stop_deadline=0;
    void run() noexcept {
        try {
            for(;;){
                feed.flush();
                std::unique_ptr<Pair> pair;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    if(stopping && stereo_monotonic_ns()>=stop_deadline){
                        if(queued){queued.reset();++replaced;}feed.abandon_pending();break;
                    }
                    if(!feed.has_pending() && queued)pair=std::move(queued);
                    else if(stopping && !feed.has_pending() && !queued)break;
                    else {
                        if(feed.has_pending())ready.wait_for(lock,std::chrono::milliseconds(2));
                        else ready.wait(lock,[&]{return stopping || bool(queued);});
                        continue;
                    }
                }
                feed.set_superseded(replaced.load());feed.set_expired(expired_pairs.load());feed.set_fixture_publish(pair->fixture_publish_ns);
                for(unsigned side=0;side<2;++side){
                    const auto& r=pair->rois[side];
                    feed.submit(pair->frame[side],pair->points[side],pair->calls[side],pair->width,pair->height,
                                r[0],r[1],r[2],r[3],pair->jpeg[side]);
                }
            }
        }catch(...){std::lock_guard<std::mutex> lock(mutex);error=std::current_exception();}
    }
public:
    explicit AsyncPairedPoseFeed(unsigned pairs,bool fixture=false,int port=18084,const std::string& source_boot="",uint64_t source_session=0)
        :feed(pairs,true,fixture,port,source_boot,source_session),fixture_mode(fixture){}
    ~AsyncPairedPoseFeed(){try{finish();}catch(...){}}
    bool wait_for_client(unsigned timeout_ms){return feed.wait_for_client(timeout_ms);}
    void submit(const CameraFrame (&frames)[2],const std::vector<float> (&points)[2],
                const std::vector<uint8_t> (&jpeg)[2],const uint64_t (&calls)[2],int width,int height,
                const std::array<std::array<double,4>,2>& rois,uint64_t fixture_publish_ns=0) {
        if(fixture_mode!=bool(fixture_publish_ns))throw std::runtime_error("Fixture publication must be explicit and cannot enter live mode");
        if(frames[0].camera_id!=0 || frames[1].camera_id!=1 || frames[0].pair_id!=frames[1].pair_id)
            throw std::runtime_error("Hybrid whole-pair identity mismatch");
        for(unsigned side=0;side<2;++side)
            if(jpeg[side].empty() || jpeg[side].size()>4*1024*1024 || points[side].size()!=51 || calls[side]!=110)
                throw std::runtime_error("Hybrid pair requires two complete original JPEG/FPGA results");
        if(jpeg[0].size()+jpeg[1].size()+16388>u02_hybrid::max_body_bytes)
            throw std::runtime_error("Hybrid JPEG pair leaves no bounded metadata room");
        std::unique_ptr<Pair> pair(new Pair);pair->width=width;pair->height=height;pair->rois=rois;pair->fixture_publish_ns=fixture_publish_ns;
        for(unsigned side=0;side<2;++side){pair->frame[side]=frames[side];pair->points[side]=points[side];pair->jpeg[side]=jpeg[side];pair->calls[side]=calls[side];}
        std::lock_guard<std::mutex> lock(mutex);
        if(error)std::rethrow_exception(error);
        if(stopping)throw std::runtime_error("Hybrid sender is stopped");
        if(queued)++replaced;
        queued=std::move(pair);
        if(!worker.joinable())worker=std::thread(&AsyncPairedPoseFeed::run,this);
        ready.notify_one();
    }
    void finish(){
        {std::lock_guard<std::mutex> lock(mutex);if(!stopping){stopping=true;stop_deadline=stereo_monotonic_ns()+500000000ull;}ready.notify_one();}
        if(worker.joinable())worker.join();
        if(error)std::rethrow_exception(error);
    }
    // Read after finish: no sender thread remains.
    uint64_t sent() const{return feed.sent();}
    uint64_t dropped() const{return feed.dropped()+replaced.load();}
    uint64_t superseded() const{return replaced.load();}
    void discard_expired_pair(){++expired_pairs;}
    uint64_t expired() const{return expired_pairs.load();}
};
