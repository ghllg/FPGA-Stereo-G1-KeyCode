#pragma once

// Admission policy only: never shift timestamps or certify exposure sync.
// The production ABI7 and original ABI9 builds retain their strict 10ms gate.
namespace u02_stereo_pair_policy {
#ifdef U02_STEREO_NEAREST_RAW_20MS
#if !defined(U02_STREAM_PIPELINE) || !defined(U02_STEREO_SOURCE_1080P) || !defined(U02_STEREO_PAIR_FIRST) || !defined(U02_FORMAL_STEREO_ABI3) || !defined(U02_1080P_STARTUP_QUALIFIED) || defined(U02_STEREO_HALF_FRAME_RAW)
#error Nearest 20ms admission requires the isolated qualified ABI9 1080p pair-first path
#endif
constexpr int limit_us=20000;
constexpr int startup_limit_us=20000;
constexpr const char* name="nearest_raw_20ms";
constexpr const char* status="software_raw_nearest_match";
#elif defined(U02_STEREO_HALF_FRAME_RAW)
#if !defined(U02_STREAM_PIPELINE) || !defined(U02_STEREO_SOURCE_1080P) || defined(U02_STEREO_STARTUP_QUALIFIED)
#error Half-frame raw admission is isolated to the explicit ABI9 1080p experiment
#endif
constexpr int limit_us=16667; // ceil(1e6 / (2 * requested 30 FPS))
constexpr const char* name="bounded_half_frame_raw";
constexpr const char* status="software_raw_half_frame_match";
#else
constexpr int limit_us=10000;
constexpr const char* name="strict_raw_10ms";
constexpr const char* status="software_raw_timestamp_match";
#endif
#ifndef U02_STEREO_NEAREST_RAW_20MS
constexpr int startup_limit_us=10000;
#endif
}
