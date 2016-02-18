LOCAL_PATH := $(call my-dir)
LOCAL_PATH_EXT := $(call my-dir)/../external/
include $(CLEAR_VARS)

LOCAL_MODULE := OpenCLTest

LOCAL_CFLAGS += -DANDROID_CL
LOCAL_CFLAGS += -O3 -ffast-math

LOCAL_C_INCLUDES := $(LOCAL_PATH)/../include

LOCAL_SRC_FILES := OpenCLTest.cpp

#LOCAL_LDFLAGS += -ljnigraphics
LOCAL_LDLIBS := -llog -ljnigraphics
LOCAL_LDLIBS += $(LOCAL_PATH)/../external/libOpenCL.so

LOCAL_ARM_MODE := arm

include $(BUILD_SHARED_LIBRARY)
