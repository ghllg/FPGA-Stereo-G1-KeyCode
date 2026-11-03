#pragma once
#include <cstddef>
#include <cstdio>
#include <csetjmp>
#include <cstdint>
#include <jpeglib.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Use the existing libjpeg-turbo8 decoder with exactly the same ISLOW DCT,
// fancy upsampling, RGB output, no scaling and guarded horizontal crop. Only
// skip conversion/storage of rows absent from the exact affine sample set.
// Progressive/lossless/non-RGB-compatible inputs retain the TurboJPEG path.
namespace u02_sparse_jpeg {
struct Context {
    jpeg_decompress_struct decoder{};
    jpeg_error_mgr error{};
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX]{};
    bool created=false;
    ~Context(){if(created)jpeg_destroy_decompress(&decoder);}
};
inline void fail(j_common_ptr decoder) {
    Context* ctx=static_cast<Context*>(decoder->client_data);
    (*decoder->err->format_message)(decoder,ctx->message);
    std::longjmp(ctx->jump,1);
}
inline void warning(j_common_ptr decoder,int level) {if(level<0)fail(decoder);}

inline bool decode(const std::vector<uint8_t>& encoded,int width,int height,
                   int& origin_x,int& stride,double iy,double oy,
                   std::vector<unsigned char>& pixels,std::vector<int>& row_index,
                   Context* reusable=nullptr,std::vector<int>* retained_rows=nullptr) {
    // Heap-owned decoder state survives longjmp. All C++ owners are created
    // before setjmp; the jump crosses only the C codec stack, not destructors.
    std::unique_ptr<Context> owner(reusable?nullptr:new Context);
    Context* ctx=reusable?reusable:owner.get();
    // Abort releases reusable image-pool state on fallback/errors; successful
    // finish already resets it. A non-reused decoder is destroyed by owner.
    // No C++ destructor is crossed by the codec's longjmp.
    struct Abort {Context* ctx;bool armed;~Abort(){if(armed && ctx->created)jpeg_abort_decompress(&ctx->decoder);}} abort{ctx,bool(reusable)};
    std::vector<int> local_rows;
    std::vector<int>& rows=retained_rows?*retained_rows:local_rows;
    rows.clear();
    row_index.assign(height,-1);
    for(int y=0;y<128;++y) {
        const int64_t first=int64_t(std::floor(iy*y+oy));
        for(int dy=0;dy<2;++dy)if(first+dy>=0 && first+dy<height)rows.push_back(int(first+dy));
    }
    std::sort(rows.begin(),rows.end());rows.erase(std::unique(rows.begin(),rows.end()),rows.end());
    ctx->decoder.err=jpeg_std_error(&ctx->error);
    ctx->error.error_exit=fail;ctx->error.emit_message=warning;
    ctx->decoder.client_data=ctx;
    if(setjmp(ctx->jump))throw std::runtime_error(std::string("Sparse JPEG: ")+ctx->message);
    if(!ctx->created){jpeg_create_decompress(&ctx->decoder);ctx->created=true;}
    jpeg_mem_src(&ctx->decoder,encoded.data(),static_cast<unsigned long>(encoded.size()));
    if(jpeg_read_header(&ctx->decoder,TRUE)!=JPEG_HEADER_OK)throw std::runtime_error("Sparse JPEG header suspended");
    if(ctx->decoder.image_width!=unsigned(width) || ctx->decoder.image_height!=unsigned(height))
        throw std::runtime_error("Sparse JPEG source dimensions differ");
    // Caller has excluded lossless JPEG using the TurboJPEG header parser.
    if(ctx->decoder.progressive_mode || ctx->decoder.data_precision!=8)
        return false;
    if(ctx->decoder.jpeg_color_space!=JCS_YCbCr && ctx->decoder.jpeg_color_space!=JCS_RGB &&
       ctx->decoder.jpeg_color_space!=JCS_GRAYSCALE)return false;
    ctx->decoder.out_color_space=JCS_RGB;
    ctx->decoder.dct_method=JDCT_ISLOW;ctx->decoder.do_fancy_upsampling=TRUE;
    if(!jpeg_start_decompress(&ctx->decoder))throw std::runtime_error("Sparse JPEG start suspended");
    JDIMENSION crop_x=origin_x,crop_w=stride;
    if(origin_x || stride!=width)jpeg_crop_scanline(&ctx->decoder,&crop_x,&crop_w);
    if(crop_x!=unsigned(origin_x) || crop_w!=unsigned(stride) || ctx->decoder.output_components!=3)
        throw std::runtime_error("Sparse JPEG crop differs from exact ROI path");
    // Keep one unmapped scratch row for validating the compressed tail.
    pixels.resize((rows.size()+1)*size_t(stride)*3);
    for(size_t i=0;i<rows.size();++i) {
        const JDIMENSION target=rows[i];
        if(ctx->decoder.output_scanline<target) {
            const JDIMENSION skip=target-ctx->decoder.output_scanline;
            if(jpeg_skip_scanlines(&ctx->decoder,skip)!=skip)throw std::runtime_error("Sparse JPEG skip suspended");
        }
        JSAMPROW line=pixels.data()+i*size_t(stride)*3;
        if(jpeg_read_scanlines(&ctx->decoder,&line,1)!=1)throw std::runtime_error("Sparse JPEG row suspended");
        row_index[target]=int(i);
    }
    const JDIMENSION remaining=ctx->decoder.output_height-ctx->decoder.output_scanline;
    if(remaining) {
        // libjpeg's skip-to-end shortcut marks EOI without reading the tail.
        // Read the final row and finish normally so truncated data is rejected
        // even when the affine ROI uses only the top or no source rows.
        if(remaining>1 && jpeg_skip_scanlines(&ctx->decoder,remaining-1)!=remaining-1)
            throw std::runtime_error("Sparse JPEG final skip suspended");
        JSAMPROW tail=pixels.data()+rows.size()*size_t(stride)*3;
        if(jpeg_read_scanlines(&ctx->decoder,&tail,1)!=1)
            throw std::runtime_error("Sparse JPEG final row suspended");
    }
    if(!jpeg_finish_decompress(&ctx->decoder))throw std::runtime_error("Sparse JPEG finish suspended");
    // finish_decompress has already released the image pool and reset state.
    abort.armed=false;
    return true;
}
}
