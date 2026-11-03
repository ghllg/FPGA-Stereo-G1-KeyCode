"""Decode a verified stereo JPEG pair on two persistent CPU workers, in source order."""
from concurrent.futures import ThreadPoolExecutor, wait

import cv2
import numpy as np


_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix='stereo-jpeg')


def _decode(data, source_size):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or list(image.shape[1::-1]) != list(source_size):
        raise ValueError('Decoded JPEG size differs from the source image size.')
    return image


def decode_pair(payloads, source_sizes):
    if (type(payloads) is not tuple or len(payloads) != 2
            or any(type(data) is not bytes for data in payloads) or len(source_sizes) != 2):
        raise ValueError('Exactly two immutable JPEG payloads and source sizes are required.')
    jobs = [_POOL.submit(_decode, data, size) for data, size in zip(payloads, source_sizes)]
    wait(jobs)
    return [job.result() for job in jobs]
