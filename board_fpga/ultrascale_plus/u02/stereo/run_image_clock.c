/* Clock the real board image Host process without changing its accepted ELF. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static uint64_t now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t))return 0;
    return (uint64_t)t.tv_sec*1000000000u+(uint64_t)t.tv_nsec;
}
int main(int argc,char **argv) {
    if(argc<3)return 2;
    uint64_t start=now();if(!start)return 3;
    pid_t child=fork();if(child<0)return 4;
    if(!child){execv(argv[2],argv+2);_exit(127);}
    int status;
    while(waitpid(child,&status,0)<0){if(errno!=EINTR)return 5;}
    uint64_t end=now();if(!end)return 6;
    FILE *file=fopen(argv[1],"wx");if(!file)return 7;
    int code=WIFEXITED(status)?WEXITSTATUS(status):128+(WIFSIGNALED(status)?WTERMSIG(status):0);
    int wrote=fprintf(file,"{\"clock\":\"CLOCK_MONOTONIC\",\"start_ns\":%"PRIu64",\"end_ns\":%"PRIu64",\"child_exit_code\":%d,\"scope\":\"process_envelope_including_model_load_and_io\"}\n",start,end,code);
    if(fclose(file) || wrote<0)return 8;
    return code;
}
