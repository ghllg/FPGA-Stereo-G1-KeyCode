obj-m := u02_uvc1080.o
u02_uvc1080-y := uvc_driver.o uvc_queue.o uvc_v4l2.o uvc_video.o uvc_ctrl.o uvc_status.o uvc_isight.o uvc_debugfs.o uvc_metadata.o
ifeq ($(CONFIG_MEDIA_CONTROLLER),y)
u02_uvc1080-y += uvc_entity.o
endif
