#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct PoseImageTimes { double decode_ms=0, preprocess_ms=0; };
std::vector<uint8_t> read_pose_image(const std::string& path);
// RGB JPEG/PNG decode, affine bilinear sampling with zero padding,
// then ImageNet normalization into the model's 3x128x96 FP32 CHW input.
PoseImageTimes prepare_pose_image(const std::vector<uint8_t>& encoded,
    int source_width, int source_height, double rx, double ry, double rw,
    double rh, float* chw);
// Same source-image affine and normalization, with guarded horizontal JPEG
// region decode. Source coordinates and declared dimensions stay unchanged.
PoseImageTimes prepare_pose_image_roi(const std::vector<uint8_t>& encoded,
    int source_width,int source_height,double rx,double ry,double rw,double rh,float* chw);

#ifdef U02_IMAGE_WORKSPACE
#include <memory>
#ifndef U02_IMAGE_WORKSPACE_REUSE_LEVEL
#define U02_IMAGE_WORKSPACE_REUSE_LEVEL 2
#endif
// One workspace per camera, used by only one decode operation at a time.
// Ownership is explicit: the left decoder currently runs on a new thread for
// each pair, so a thread_local cache would disappear after every left image.
class PoseImageWorkspace {
public:
    struct Impl;
    std::unique_ptr<Impl> impl;
    // 0: sampling plan only; 1: also buffers; 2: also JPEG decoder state.
    explicit PoseImageWorkspace(int reuse_level=U02_IMAGE_WORKSPACE_REUSE_LEVEL);
    ~PoseImageWorkspace();
    PoseImageWorkspace(const PoseImageWorkspace&)=delete;
    PoseImageWorkspace& operator=(const PoseImageWorkspace&)=delete;
};
PoseImageTimes prepare_pose_image_roi(const std::vector<uint8_t>& encoded,
    int source_width,int source_height,double rx,double ry,double rw,double rh,
    float* chw,PoseImageWorkspace& workspace);
#endif
