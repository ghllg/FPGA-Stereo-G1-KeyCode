#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct CameraFrame {
    uint32_t sequence=0, flags=0, discarded_ready_frames=0;
    uint64_t timestamp_us=0;
    double capture_ms=0;
    uint32_t camera_id=0;
    uint64_t pair_id=0, receive_monotonic_ns=0, infer_start_ns=0, infer_end_ns=0;
    uint64_t dequeue_monotonic_ns=0;
    int64_t pair_delta_us=0;
    uint64_t dropped_capacity=0, dropped_age=0, dropped_mismatch=0, dropped_superseded=0;
};

// Bounded, single-camera MJPEG capture. Driver timestamps are not treated as
// confirmed exposure times. Owns STREAMOFF, mappings and fd on every exit.
class CameraCapture {
public:
    CameraCapture(const std::string& node,int width,int height,bool start_immediately=true);
    ~CameraCapture();
    CameraFrame next(std::vector<uint8_t>& jpeg);
    void start();
    void stop();
    const std::string& bus_id() const;
private:
    struct State;
    std::unique_ptr<State> state;
};
