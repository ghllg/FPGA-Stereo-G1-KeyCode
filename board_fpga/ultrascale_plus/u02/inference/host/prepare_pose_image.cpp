#include "image_preprocess.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
int main(int argc,char** argv) {
    try {
        if(argc!=9)throw std::runtime_error("Usage: prepare_pose_image IMAGE OUTPUT.f32 SOURCE_W SOURCE_H ROI_X ROI_Y ROI_W ROI_H");
        const auto encoded=read_pose_image(argv[1]);
        std::vector<float> input(3*128*96);
        const auto times=prepare_pose_image(encoded,std::stoi(argv[3]),std::stoi(argv[4]),
            std::stod(argv[5]),std::stod(argv[6]),std::stod(argv[7]),std::stod(argv[8]),input.data());
        std::ofstream f(argv[2],std::ios::binary);
        f.write(reinterpret_cast<const char*>(input.data()),input.size()*sizeof(float));f.close();
        if(!f)throw std::runtime_error("Normalized input write failed");
        const char* decoder=
#ifdef U02_IMAGE_JPEG_TURBO
            "libjpeg-turbo 3.1.2 JPEG / stb PNG";
#else
            "stb_image RGB JPEG/PNG";
#endif
        std::cout<<std::setprecision(10)<<"{\"complete\":true,\"decoder\":\""<<decoder<<"\",\"input_shape\":[3,128,96],\"decode_ms\":"
            <<times.decode_ms<<",\"roi_normalize_ms\":"<<times.preprocess_ms<<",\"scope\":\"decode_and_preprocess_only_excludes_file_IO_and_inference\"}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
