#pragma once
#include "pose_network_session.h"
#include <thread>
#include <condition_variable>
#include <exception>

// One persistent worker per private session/input. The caller may replace input
// pixels only after wait() completes and before the next submit().
class PoseWorker {
    PoseNetworkSession& session;
    const ncnn::Mat& input;
    std::mutex mutex;
    std::condition_variable ready,finished;
    bool requested=false,busy=false,stopping=false;
    std::exception_ptr error;
    std::thread worker;
    void loop() {
        for(;;) {
            std::unique_lock<std::mutex> lock(mutex);
            ready.wait(lock,[&]{return requested || stopping;});
            if(stopping && !requested)return;
            requested=false;lock.unlock();
            std::exception_ptr failure;
            try {session.forward(input);}catch(...) {failure=std::current_exception();}
            lock.lock();if(failure)error=failure;busy=false;finished.notify_one();
        }
    }
public:
    PoseWorker(PoseNetworkSession& s,const ncnn::Mat& in):session(s),input(in),worker([this]{loop();}) {}
    ~PoseWorker() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;ready.notify_one();}
        worker.join();
    }
    void submit() {
        std::lock_guard<std::mutex> lock(mutex);
        if(error)std::rethrow_exception(error);
        if(busy || stopping)throw std::runtime_error("Worker already busy or stopped");
        busy=requested=true;ready.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);finished.wait(lock,[&]{return !busy;});
        if(error)std::rethrow_exception(error);
    }
};
