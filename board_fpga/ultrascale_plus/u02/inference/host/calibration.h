#pragma once
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

// Observations from actual full-network forwards, never performance samples.
// DW consumes per-channel scales; ordinary convolution consumes one scale.
struct CalibrationRecord {
    std::string layer;
    int channels, height, width;
    std::vector<float> maxima, scales;
    void write(std::ostream& out) const {
        if(layer.empty() || layer.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.")!=std::string::npos)
            throw std::runtime_error("Unsupported calibration layer name");
        out << "{\"layer\":\"" << layer << "\",\"input_shape\":[1," << channels << ',' << height << ',' << width
            << "],\"granularity\":\"" << (scales.size()==1?"tensor":"channel") << "\",\"max_abs_per_channel\":[";
        for(size_t i=0;i<maxima.size();++i) out << (i?",":"") << maxima[i];
        out << "],\"dynamic_scales\":[";
        for(size_t i=0;i<scales.size();++i) out << (i?",":"") << scales[i];
        out << "]}";
    }
};
