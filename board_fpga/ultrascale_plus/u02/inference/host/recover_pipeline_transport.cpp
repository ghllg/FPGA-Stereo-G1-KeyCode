// Recovery only, after external bit/boot-bound JTAG idle proof. No mmap/RUN.
#include "../../board/linux/u02_stream_pipeline_api.h"
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
int main(int argc,char** argv) {
    if(argc!=2 || std::strcmp(argv[1],"--recover-after-independent-idle"))return 2;
    const int fd=open("/dev/u02_stream_pipeline",O_RDWR|O_CLOEXEC);
    if(fd<0){std::perror("open recovery owner");return 1;}
    u02_pose_info info{};
    if(ioctl(fd,U02_POSE_INFO,&info) || info.abi!=9 || info.arena_bytes!=U02_POSE_ARENA_BYTES) {
        std::fprintf(stderr,"Recovery ABI/capacity mismatch\n");close(fd);return 1;
    }
    const int rc=ioctl(fd,U02_POSE_RECOVER);
    if(rc)std::perror("RECOVER requires actual idle; retain DMA/module ownership");
    close(fd);
    if(!rc)std::puts("{\"complete\":true,\"scope\":\"kernel_RECOVER_after_external_idle_proof\",\"run_submissions\":0,\"dma_mapped\":false}");
    return rc?1:0;
}
