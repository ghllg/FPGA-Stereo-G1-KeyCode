#pragma once

// Capture geometry is a build-time choice. The verified default stays 720p;
// a separate executable explicitly opts into 1080p. Neither changes the
// PP-TinyPose network tensor (3 x 128 x 96).
namespace u02_stereo_source {
#ifdef U02_STEREO_SOURCE_1080P
#if !defined(U02_FORMAL_STEREO_ABI3) || !defined(U02_STEREO_PAIR_FIRST) || defined(U02_STEREO_STARTUP_QUALIFIED)
#error The 1080p candidate requires the formal pair-first path, without frozen startup retry experiments
#endif
constexpr int width=1920, height=1080;
#else
constexpr int width=1280, height=720;
#endif
}
