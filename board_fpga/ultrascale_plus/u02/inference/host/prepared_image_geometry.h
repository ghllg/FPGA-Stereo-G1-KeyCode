#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Exact affine sampling plan. Keep the FP64 multiply/accumulate order of the
// accepted implementation; pre-multiplying bilinear weights can change uint8
// rounding at half-integer boundaries.
struct PreparedImageGeometry {
    int width=0,height=0,origin=0,stride=0;
    double ix=0,iy=0,ox=0,oy=0;
    bool sparse=false,ready=false;
    int xoffset[96][2],yoffset[128][2];
    double wx[96][2],wy[128][2];
    float normalized[3][256];
    PreparedImageGeometry() {
        const float mean[3]={.485f,.456f,.406f},dev[3]={.229f,.224f,.225f};
        for(int c=0;c<3;++c)for(int p=0;p<256;++p)
            normalized[c][p]=(float(p)/255.0f-mean[c])/dev[c];
    }
    void prepare(int w,int h,int base,int pitch,double xstep,double ystep,
                 double xstart,double ystart,const std::vector<int>* rows) {
        if(ready && width==w && height==h && origin==base && stride==pitch &&
           ix==xstep && iy==ystep && ox==xstart && oy==ystart && sparse==bool(rows))return;
        ready=false;
        for(int x=0;x<96;++x) {
            const double at=xstep*x+xstart;const int64_t first=int64_t(std::floor(at));
            wx[x][1]=at-first;wx[x][0]=1-wx[x][1];
            for(int dx=0;dx<2;++dx) {
                const int64_t px=first+dx;xoffset[x][dx]=-1;
                if(px>=0 && px<w) {
                    if(px<base || px>=base+pitch)
                        throw std::runtime_error("JPEG ROI region does not cover affine sample");
                    xoffset[x][dx]=int(px-base)*3;
                }
            }
        }
        for(int y=0;y<128;++y) {
            const double at=ystep*y+ystart;const int64_t first=int64_t(std::floor(at));
            wy[y][1]=at-first;wy[y][0]=1-wy[y][1];
            for(int dy=0;dy<2;++dy) {
                const int64_t py=first+dy;yoffset[y][dy]=-1;
                if(py>=0 && py<h) {
                    const int stored=rows?rows->at(size_t(py)):int(py);
                    if(stored<0)throw std::runtime_error("Sparse JPEG row missing affine sample");
                    yoffset[y][dy]=stored*pitch*3;
                }
            }
        }
        width=w;height=h;origin=base;stride=pitch;
        ix=xstep;iy=ystep;ox=xstart;oy=ystart;sparse=bool(rows);ready=true;
    }
    void sample(const unsigned char* pixels,float* chw) const {
        for(int y=0;y<128;++y)for(int x=0;x<96;++x) {
            double r=0,g=0,b=0;
            for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx) {
                if(yoffset[y][dy]<0 || xoffset[x][dx]<0)continue;
                const auto* p=pixels+yoffset[y][dy]+xoffset[x][dx];
                r+=p[0]*wx[x][dx]*wy[y][dy];
                g+=p[1]*wx[x][dx]*wy[y][dy];
                b+=p[2]*wx[x][dx]*wy[y][dy];
            }
            const int at=y*96+x;
            chw[at]=normalized[0][int(std::nearbyint(r))];
            chw[128*96+at]=normalized[1][int(std::nearbyint(g))];
            chw[2*128*96+at]=normalized[2][int(std::nearbyint(b))];
        }
    }
};
