// Sample at the current input time, never select a future source frame.
export function limitTo12FPS(clip) {
  if (clip.dt >= 1 / 12) return clip;
  const dt = 1 / 12;
  const count = Math.ceil(clip.frames.length * clip.dt / dt - 1e-9);
  const frames = Array.from({length: count}, (_, i) =>
    clip.frames[Math.min(clip.frames.length - 1, Math.floor(i * dt / clip.dt + 1e-9))]);
  return {...clip, dt, frames};
}
