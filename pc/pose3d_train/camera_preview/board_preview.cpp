// Live framing preview; uses the handoff package's unchanged V4L2 capture code.
#include "camera_capture.h"
#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

namespace {
volatile sig_atomic_t stopping = 0;
void stop(int) { stopping = 1; }
struct Socket {
    int fd = -1;
    ~Socket() { if (fd >= 0) close(fd); }
};
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
bool send_all(int fd, const void* data, size_t size) {
    const char* p = static_cast<const char*>(data);
    while (size && !stopping) {
        const ssize_t n = send(fd, p, size, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        size -= n;
    }
    return size == 0;
}
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::puts("Dual framing preview: TCP 18085, video0=USB 1.4, video2=USB 1.2.");
        std::puts("Cameras start only after a client connects. --help opens no camera.");
        return 0;
    }
    if (argc != 1) return 2;
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    std::signal(SIGPIPE, SIG_IGN);
    try {
        Socket listener;
        listener.fd = socket(AF_INET, SOCK_STREAM, 0);
        require(listener.fd >= 0, "socket failed");
        int one = 1;
        setsockopt(listener.fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in address = {};
        address.sin_family = AF_INET;
        address.sin_port = htons(18085);
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        require(bind(listener.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
                "Preview port 18085 is occupied");
        require(listen(listener.fd, 1) == 0, "listen failed");
        std::puts("LISTENING 18085 (waiting for PC; cameras closed)");
        std::fflush(stdout);
        pollfd ready = {listener.fd, POLLIN, 0};
        if (poll(&ready, 1, 30000) <= 0 || stopping) return 0;
        Socket client;
        client.fd = accept(listener.fd, nullptr, nullptr);
        require(client.fd >= 0, "accept failed");
        timeval timeout = {2, 0};
        require(setsockopt(client.fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0,
                "send timeout setup failed");
        int buffer_bytes = 256 * 1024;
        setsockopt(client.fd, SOL_SOCKET, SO_SNDBUF, &buffer_bytes, sizeof(buffer_bytes));
        CameraCapture left("/dev/video0", 1280, 720);
        CameraCapture right("/dev/video2", 1280, 720);
        require(left.bus_id() == "usb-xhci-hcd.0.auto-1.4" &&
                right.bus_id() == "usb-xhci-hcd.0.auto-1.2", "Physical camera identities changed");
        std::puts("STREAMING left=USB1.4 right=USB1.2; framing only, not synchronized pairs");
        std::fflush(stdout);
        CameraCapture* cameras[] = {&left, &right};
        std::vector<uint8_t> jpeg;
        while (!stopping) {
            for (uint32_t id = 0; id < 2 && !stopping; ++id) {
                const CameraFrame frame = cameras[id]->next(jpeg);
                // Network byte order: magic, camera, sequence, JPEG bytes, timestamp hi/lo.
                uint32_t header[] = {htonl(0x44565031), htonl(id), htonl(frame.sequence),
                    htonl(static_cast<uint32_t>(jpeg.size())),
                    htonl(static_cast<uint32_t>(frame.timestamp_us >> 32)),
                    htonl(static_cast<uint32_t>(frame.timestamp_us))};
                if (!send_all(client.fd, header, sizeof(header)) ||
                    !send_all(client.fd, jpeg.data(), jpeg.size())) stopping = 1;
            }
        }
        left.stop();
        right.stop();
        std::puts("STOPPED: both cameras closed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "PREVIEW_ERROR: %s\n", e.what());
        return 1;
    }
}
