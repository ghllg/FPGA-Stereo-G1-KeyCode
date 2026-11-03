#pragma once
#include <layer.h>
ncnn::Layer* create_stream(void* userdata);
const char* stream_model_plan_sha256();
