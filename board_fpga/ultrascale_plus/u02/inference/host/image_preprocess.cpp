#include "image_preprocess.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_MAX_DIMENSIONS 4096
#include "stb_image.h"
#ifdef U02_IMAGE_JPEG_TURBO
#include <turbojpeg.h>
#ifdef U02_IMAGE_SPARSE_ROWS
#include "sparse_jpeg_rows.h"
#endif
#endif
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#ifdef U02_IMAGE_WORKSPACE
#include "prepared_image_geometry.h"
struct PoseImageWorkspace::Impl {
    const int reuse_level;
    explicit Impl(int level):reuse_level(level) {
        if(level<0 || level>2)throw std::runtime_error("Invalid image workspace reuse level");
    }
    PreparedImageGeometry geometry;
    std::vector<unsigned char> pixels;
#ifdef U02_IMAGE_SPARSE_ROWS
    u02_sparse_jpeg::Context sparse_decoder;
    std::vector<int> row_index,rows;
#endif
};
PoseImageWorkspace::PoseImageWorkspace(int level):impl(new Impl(level)){}
PoseImageWorkspace::~PoseImageWorkspace()=default;
#endif

std::vector<uint8_t> read_pose_image(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    const auto bytes=f.tellg();
    if(!f || bytes<=0 || bytes>16*1024*1024)
        throw std::runtime_error("Expected an image file between 1 byte and 16 MiB");
    std::vector<uint8_t> result(static_cast<size_t>(bytes));f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(result.data()),result.size()))
        throw std::runtime_error("Image read failed");
    return result;
}
static PoseImageTimes prepare_internal(const std::vector<uint8_t>& encoded,
    int source_width,int source_height,double rx,double ry,double rw,double rh,float* chw,bool region_decode
#ifdef U02_IMAGE_WORKSPACE
    ,PoseImageWorkspace::Impl* workspace=nullptr
#endif
    ) {
    if(!chw || encoded.empty() || encoded.size()>16*1024*1024 ||
       source_width<1 || source_height<1 || source_width>4096 || source_height>4096 ||
       !std::isfinite(rx+ry+rw+rh) || rw<.001 || rh<.001 ||
       std::fabs(rx)>1000000 || std::fabs(ry)>1000000 || rw>1000000 || rh>1000000)
        throw std::runtime_error("Invalid image or affine ROI geometry");
    using Clock=std::chrono::steady_clock;
    const auto start=Clock::now();int width=0,height=0,channels=0;
    int pixel_origin_x=0,pixel_stride=source_width;
    const double a=float(96.0/rw),b=float(128.0/rh);
    const double tx=float(-rx*96.0/rw),ty=float(-ry*128.0/rh);
    const double determinant=1.0/(a*b),ix=b*determinant,iy=a*determinant;
    const double ox=-ix*tx,oy=-iy*ty;
    std::unique_ptr<unsigned char,void(*)(void*)> rgb(nullptr,stbi_image_free);
    std::vector<unsigned char> local_rgb;
#ifdef U02_IMAGE_WORKSPACE
    auto& jpeg_rgb=workspace && workspace->reuse_level>=1?workspace->pixels:local_rgb;
#else
    auto& jpeg_rgb=local_rgb;
#endif
#ifdef U02_IMAGE_SPARSE_ROWS
    std::vector<int> local_row_index;
#ifdef U02_IMAGE_WORKSPACE
    auto& jpeg_row_index=workspace && workspace->reuse_level>=1?workspace->row_index:local_row_index;
#else
    auto& jpeg_row_index=local_row_index;
#endif
    bool sparse_rows=false;
#endif
    const unsigned char* pixels=nullptr;
#ifdef U02_IMAGE_JPEG_TURBO
    if(encoded.size()>=2 && encoded[0]==0xff && encoded[1]==0xd8) {
        std::unique_ptr<void,void(*)(tjhandle)> decoder(tj3Init(TJINIT_DECOMPRESS),tj3Destroy);
        if(!decoder)throw std::runtime_error("JPEG decoder initialization failed");
        if(tj3Set(decoder.get(),TJPARAM_MAXPIXELS,4096*4096) ||
           tj3Set(decoder.get(),TJPARAM_STOPONWARNING,1) ||
           tj3Set(decoder.get(),TJPARAM_FASTDCT,0) ||
           tj3Set(decoder.get(),TJPARAM_FASTUPSAMPLE,0) ||
           tj3DecompressHeader(decoder.get(),encoded.data(),encoded.size()))
            throw std::runtime_error(tj3GetErrorStr(decoder.get()));
        width=tj3Get(decoder.get(),TJPARAM_JPEGWIDTH);height=tj3Get(decoder.get(),TJPARAM_JPEGHEIGHT);
        if(width!=source_width || height!=source_height)
            throw std::runtime_error("Decoded dimensions differ from declared source geometry");
        pixel_stride=width;
        const int subsamp=tj3Get(decoder.get(),TJPARAM_SUBSAMP);
        if(region_decode && !tj3Get(decoder.get(),TJPARAM_LOSSLESS) && subsamp>=0 && subsamp<TJ_NUMSAMP) {
            const int unit=tjMCUWidth[subsamp];
            // Keep an extra iMCU on both sides for fancy chroma upsampling.
            // Bounds derive from the exact existing FP32 affine inverse.
            const int64_t first=int64_t(std::floor(ox));
            const int64_t last=int64_t(std::floor(ix*95+ox))+1;
            if(last>=0 && first<width) {
                const int x0=int(std::max<int64_t>(0,first-unit)/unit)*unit;
                const int x1=int(std::min<int64_t>(width,((std::min<int64_t>(width-1,last)+unit+1+unit-1)/unit)*unit));
                if(x0>0 || x1<width) {
                    tjregion region={x0,0,x1-x0,height};
                    if(region.w<=0 || tj3SetCroppingRegion(decoder.get(),region))
                        throw std::runtime_error("JPEG ROI region configuration failed");
                    pixel_origin_x=x0;pixel_stride=region.w;
                }
            }
        }
#ifdef U02_IMAGE_SPARSE_ROWS
        if(region_decode && !tj3Get(decoder.get(),TJPARAM_LOSSLESS))
            sparse_rows=u02_sparse_jpeg::decode(encoded,width,height,pixel_origin_x,pixel_stride,iy,oy,jpeg_rgb,jpeg_row_index
#ifdef U02_IMAGE_WORKSPACE
                ,workspace && workspace->reuse_level>=2?&workspace->sparse_decoder:nullptr,
                workspace && workspace->reuse_level>=1?&workspace->rows:nullptr
#endif
            );
        if(!sparse_rows)
#endif
        {
            jpeg_rgb.resize(size_t(pixel_stride)*height*3);
            if(tj3Decompress8(decoder.get(),encoded.data(),encoded.size(),jpeg_rgb.data(),0,TJPF_RGB))
                throw std::runtime_error(tj3GetErrorStr(decoder.get()));
        }
        pixels=jpeg_rgb.data();
    } else
#endif
    {
        rgb.reset(stbi_load_from_memory(encoded.data(),static_cast<int>(encoded.size()),
                                       &width,&height,&channels,3));
        if(!rgb) throw std::runtime_error("JPEG/PNG decode failed");
        pixels=rgb.get();
    }
    if(width!=source_width || height!=source_height)
        throw std::runtime_error("Decoded dimensions differ from declared source geometry");
    const auto decoded=Clock::now();
#ifdef U02_IMAGE_WORKSPACE
    if(workspace) {
        const std::vector<int>* rows=nullptr;
#ifdef U02_IMAGE_SPARSE_ROWS
        if(sparse_rows)rows=&jpeg_row_index;
#endif
        workspace->geometry.prepare(width,height,pixel_origin_x,pixel_stride,ix,iy,ox,oy,rows);
        workspace->geometry.sample(pixels,chw);
        const auto done=Clock::now();PoseImageTimes times;
        times.decode_ms=std::chrono::duration<double,std::milli>(decoded-start).count();
        times.preprocess_ms=std::chrono::duration<double,std::milli>(done-decoded).count();
        return times;
    }
#endif
    // Reuse the float32 affine matrix in prepare_samples.py. Sample its actual
    // inverse coordinates rather than introducing a 1/32-pixel coordinate grid.
    int64_t source_x[96];double fraction_x[96];
    for(int x=0;x<96;++x) {
        const double at=ix*x+ox;source_x[x]=int64_t(std::floor(at));
        fraction_x[x]=at-source_x[x];
    }
    // Input samples are rounded to uint8 before normalization. Reuse exactly
    // the same scalar FP32 expression for each of its 256 possible values.
    // This avoids repeating three divisions at every output pixel.
    const float mean[3]={.485f,.456f,.406f},dev[3]={.229f,.224f,.225f};
    float normalized[3][256];
    for(int c=0;c<3;++c)for(int pixel=0;pixel<256;++pixel)
        normalized[c][pixel]=(float(pixel)/255.0f-mean[c])/dev[c];
    for(int y=0;y<128;++y) {
        const double at_y=iy*y+oy;const int64_t sy=int64_t(std::floor(at_y));
        const double fy=at_y-sy;
        for(int x=0;x<96;++x) {
            const int64_t sx=source_x[x];const double fx=fraction_x[x];
            // Geometry and boundary checks are shared by all RGB channels.
            // Keep the original multiplication and accumulation order so
            // rounding at the uint8 boundary remains unchanged.
            const unsigned char* neighbors[4]={nullptr,nullptr,nullptr,nullptr};
            for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx) {
                const int64_t px=sx+dx,py=sy+dy;
                if(px>=0 && px<width && py>=0 && py<height) {
                    if(px<pixel_origin_x || px>=pixel_origin_x+pixel_stride)
                        throw std::runtime_error("JPEG ROI region does not cover affine sample");
                    int64_t stored_y=py;
#ifdef U02_IMAGE_SPARSE_ROWS
                    if(sparse_rows) {
                        stored_y=jpeg_row_index[py];
                        if(stored_y<0)throw std::runtime_error("Sparse JPEG row missing affine sample");
                    }
#endif
                    neighbors[dy*2+dx]=pixels+(stored_y*pixel_stride+px-pixel_origin_x)*3;
                }
            }
            for(int c=0;c<3;++c) {
                double sum=0;
                for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx) {
                    const unsigned char* neighbor=neighbors[dy*2+dx];
                    if(neighbor)
                        sum+=neighbor[c]*(dx?fx:1-fx)*(dy?fy:1-fy);
                }
                const int pixel=int(std::nearbyint(sum));
                chw[c*128*96+y*96+x]=normalized[c][pixel];
            }
        }
    }
    const auto done=Clock::now();PoseImageTimes times;
    times.decode_ms=std::chrono::duration<double,std::milli>(decoded-start).count();
    times.preprocess_ms=std::chrono::duration<double,std::milli>(done-decoded).count();
    return times;
}

PoseImageTimes prepare_pose_image(const std::vector<uint8_t>& encoded,
    int w,int h,double rx,double ry,double rw,double rh,float* chw) {
    return prepare_internal(encoded,w,h,rx,ry,rw,rh,chw,false);
}

PoseImageTimes prepare_pose_image_roi(const std::vector<uint8_t>& encoded,
    int w,int h,double rx,double ry,double rw,double rh,float* chw) {
    return prepare_internal(encoded,w,h,rx,ry,rw,rh,chw,true);
}
#ifdef U02_IMAGE_WORKSPACE
PoseImageTimes prepare_pose_image_roi(const std::vector<uint8_t>& encoded,
    int w,int h,double rx,double ry,double rw,double rh,float* chw,PoseImageWorkspace& workspace) {
    return prepare_internal(encoded,w,h,rx,ry,rw,rh,chw,true,workspace.impl.get());
}
#endif
