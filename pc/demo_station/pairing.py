"""One-to-one nearest neighbours on original, increasing camera timestamps."""
from bisect import bisect_left


def mutual_nearest_pairs(left, right, limit_us):
    if not 0 < limit_us <= 20000:
        raise ValueError('Pairing limit must be within 1–20000 microseconds')
    if any(b <= a for values in (left, right) for a, b in zip(values, values[1:])):
        raise ValueError('Camera timestamps must strictly increase')
    if not left or not right:
        return []

    def nearest(values, stamp):
        index = bisect_left(values, stamp)
        candidates = range(max(0, index-1), min(len(values), index+1))
        return min(candidates, key=lambda i: (abs(values[i]-stamp), i))

    lr = [nearest(right, stamp) for stamp in left]
    rl = [nearest(left, stamp) for stamp in right]
    return [(i, j, right[j]-left[i]) for i, j in enumerate(lr)
            if rl[j] == i and abs(right[j]-left[i]) <= limit_us]
