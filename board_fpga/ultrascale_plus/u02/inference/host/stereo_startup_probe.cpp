#include "stereo_capture.h"
#include <fstream>
#include <iostream>

// Real capture and decode only: no accelerator headers, driver or model calls.
int main(int argc,char** argv) {
    try {
        if(argc!=3 && argc!=4)throw std::runtime_error("Usage: stereo_startup_probe left_node,right_node output.json [--exercise-retry]");
        const bool exercise_retry=argc==4 && std::string(argv[3])=="--exercise-retry";
        if(argc==4 && !exercise_retry)throw std::runtime_error("Unknown probe option");
        const auto started=stereo_monotonic_ns();
        StereoCapture camera(argv[1],1280,720,370,0,540,720,exercise_retry);
        std::ofstream out(argv[2]);
        if(!out)throw std::runtime_error("Cannot write probe result");
        out<<"{\"pl_calls\":0,\"timestamp_adjustment_us\":0,\"pairs\":[";
        std::vector<uint8_t> jpeg;
        for(unsigned i=0;i<8;++i) {
            const auto left=camera.next(jpeg),right=camera.next(jpeg);
            const auto delta=int64_t(right.timestamp_us)-int64_t(left.timestamp_us);
            if(left.camera_id!=0 || right.camera_id!=1 || left.pair_id!=right.pair_id || delta>10000 || delta< -10000)
                throw std::runtime_error("Probe pair identity or original timestamp gate failed");
            if(i)out<<",";
            out<<"{\"left_us\":"<<left.timestamp_us<<",\"right_us\":"<<right.timestamp_us
               <<",\"delta_us\":"<<delta<<",\"pair_id\":"<<left.pair_id<<"}";
        }
        out<<"],\"elapsed_ns\":"<<stereo_monotonic_ns()-started;
        camera.write_timing_diagnostic(out);camera.stop();out<<"}\n";
        out.flush();if(!out)throw std::runtime_error("Probe result write failed");
        std::cout<<"U02_STARTUP_PROBE_OK actual_pairs=8 pl_calls=0"<<std::endl;
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
