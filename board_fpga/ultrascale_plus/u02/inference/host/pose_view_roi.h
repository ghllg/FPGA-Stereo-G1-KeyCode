#pragma once
#include <array>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

inline std::array<std::array<double,4>,2> pose_view_rois(
        int width,int height,double x,double y,double w,double h) {
    std::array<std::array<double,4>,2> result{{{{x,y,w,h}},{{x,y,w,h}}}};
    const char* values[2]={std::getenv("POSE_LEFT_ROI"),std::getenv("POSE_RIGHT_ROI")};
    if(bool(values[0])!=bool(values[1]))
        throw std::runtime_error("Set both POSE_LEFT_ROI and POSE_RIGHT_ROI");
    if(!values[0])return result;
    for(unsigned side=0;side<2;++side) {
        auto& r=result[side];std::istringstream input(values[side]);char comma;
        for(unsigned i=0;i<4;++i) {
            if(!(input>>r[i]) || !std::isfinite(r[i]) ||
               (i<3 && (!(input>>comma) || comma!=',')))
                throw std::runtime_error("ROI must contain four finite comma-separated numbers");
        }
        input>>std::ws;
        if(!input.eof() || r[0]<0 || r[1]<0 || r[2]<=0 || r[3]<=0 ||
           r[0]+r[2]>width || r[1]+r[3]>height)
            throw std::runtime_error("Per-view ROI must be inside the source image");
    }
    return result;
}
