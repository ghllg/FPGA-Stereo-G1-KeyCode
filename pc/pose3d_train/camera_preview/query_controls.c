/* Read V4L2 control descriptions and values. Never starts capture or writes a control. */
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: query_controls /dev/videoN\n");
        return 2;
    }
    int fd = open(argv[1], O_RDONLY | O_NONBLOCK);
    if (fd < 0) { perror("open"); return 1; }
    struct v4l2_capability cap = {0};
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
        perror("VIDIOC_QUERYCAP"); close(fd); return 1;
    }
    printf("DEVICE %s driver=%s card=%s bus=%s\n", argv[1], cap.driver, cap.card, cap.bus_info);
    struct v4l2_queryctrl query = {0};
    query.id = V4L2_CTRL_FLAG_NEXT_CTRL;
    unsigned count = 0;
    while (ioctl(fd, VIDIOC_QUERYCTRL, &query) == 0) {
        if (!(query.flags & V4L2_CTRL_FLAG_DISABLED)) {
            printf("CONTROL id=0x%08x name=%s type=%u min=%d max=%d step=%d default=%d flags=0x%x",
                   query.id, query.name, query.type, query.minimum, query.maximum,
                   query.step, query.default_value, query.flags);
            struct v4l2_control value = {0}; value.id = query.id;
            if (query.type != V4L2_CTRL_TYPE_CTRL_CLASS && query.type != V4L2_CTRL_TYPE_BUTTON
                    && ioctl(fd, VIDIOC_G_CTRL, &value) == 0)
                printf(" current=%d", value.value);
            printf("\n");
            if (query.type == V4L2_CTRL_TYPE_MENU || query.type == V4L2_CTRL_TYPE_INTEGER_MENU) {
                for (int index = query.minimum; index <= query.maximum; ++index) {
                    struct v4l2_querymenu menu = {0}; menu.id = query.id; menu.index = index;
                    if (ioctl(fd, VIDIOC_QUERYMENU, &menu) == 0) {
                        if (query.type == V4L2_CTRL_TYPE_MENU)
                            printf("MENU id=0x%08x index=%d name=%s\n", query.id, index, menu.name);
                        else printf("MENU id=0x%08x index=%d value=%lld\n", query.id, index, (long long)menu.value);
                    }
                }
            }
            ++count;
        }
        query.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
    }
    int saved_errno = errno;
    close(fd);
    if (saved_errno != EINVAL) { errno = saved_errno; perror("VIDIOC_QUERYCTRL"); return 1; }
    printf("READ_ONLY_COMPLETE controls=%u capture_started=false controls_written=false\n", count);
    return 0;
}
