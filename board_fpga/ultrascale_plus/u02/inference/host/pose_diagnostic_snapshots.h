#pragma once
#include "camera_capture.h"
#include "mat.h"
#include <array>
#include <fstream>
#include <iomanip>
#include <vector>
#include <stdexcept>

// Bounded diagnostic sampling only. Disk writes happen after capture stops.
struct PoseDiagnosticSnapshots {
    struct View {
        CameraFrame frame;
        ncnn::Mat input,heat;
        std::vector<uint8_t> jpeg;
        std::vector<float> points;
    };
    std::vector<std::array<View,2>> pairs;
    uint64_t next_ns=0;
    bool enabled;
    explicit PoseDiagnosticSnapshots(bool active):enabled(active) {}
    bool due(uint64_t now) const {return enabled && pairs.size()<20 && now>=next_ns;}
    void add(uint64_t now,const View& left,const View& right) {
        if(!due(now))return;
        if(left.frame.pair_id!=right.frame.pair_id || left.frame.camera_id!=0 || right.frame.camera_id!=1)
            throw std::runtime_error("Diagnostic snapshot pair identity mismatch");
        pairs.push_back({left,right});
        for(auto& view:pairs.back()) {
            view.input=view.input.clone();view.heat=view.heat.clone();
            if(view.input.empty() || view.heat.empty())throw std::runtime_error("Snapshot clone failed");
        }
        next_ns=now+1000000000ull;
    }
    void save(const std::string& prefix,const std::array<std::array<double,4>,2>& rois) const {
        for(size_t i=0;i<pairs.size();++i)for(unsigned side=0;side<2;++side) {
            const auto& v=pairs[i][side];const auto& r=rois[side];
            const std::string path=prefix+".snapshot_"+std::to_string(i)+(side?".right.json":".left.json");
            std::ofstream meta(path),input(path+".input.f32",std::ios::binary),heat(path+".heatmap",std::ios::binary),image(path+".jpg",std::ios::binary);
            meta<<std::setprecision(9)<<"{\"scope\":\"sampled_complete_pair\",\"camera_id\":\""<<(side?"right":"left")
                <<"\",\"pair_id\":"<<v.frame.pair_id<<",\"frame_id\":"<<v.frame.sequence<<",\"timestamp_us\":"<<v.frame.timestamp_us
                <<",\"roi\":["<<r[0]<<","<<r[1]<<","<<r[2]<<","<<r[3]<<"],\"keypoints\":[";
            for(unsigned j=0;j<17;++j)meta<<(j?",":"")<<"["<<v.points[j*3]<<","<<v.points[j*3+1]<<","<<v.points[j*3+2]<<"]";
            meta<<"]}\n";
            for(unsigned c=0;c<3;++c)input.write(reinterpret_cast<const char*>(static_cast<const float*>(v.input.channel(c))),96*128*4);
            for(unsigned c=0;c<17;++c)heat.write(reinterpret_cast<const char*>(static_cast<const float*>(v.heat.channel(c))),24*32*4);
            image.write(reinterpret_cast<const char*>(v.jpeg.data()),v.jpeg.size());
            if(!meta || !input || !heat || !image)throw std::runtime_error("Diagnostic snapshot write failed");
        }
    }
};
