"""Explicit U02H candidate: a measured FPGA pair and its exact original JPEGs."""
import hashlib
from dataclasses import dataclass, field
import json
import socket
import struct
import time
import traceback
import zlib

from .pair20_performance_receiver import ClockProbeTracker, validate_message

CHUNK_BYTES = 65512
MAX_BODY = 8 * 1024 * 1024
MAX_IMAGE = 4 * 1024 * 1024


def image_identity(item):
    return tuple(item[key] for key in ('side','path','boot_id','session_id','pair_id','frame_id','capture_timestamp_us')) + (tuple(item['source_size']),)


@dataclass(frozen=True)
class ReceivedJpeg:
    """Immutable bytes and evidence identity, hashed once before publication."""
    data: bytes
    identity: tuple
    sha256: str = field(init=False)

    def __post_init__(self):
        if type(self.data) is not bytes or type(self.identity) is not tuple:
            raise ValueError('Received JPEG payload must be immutable.')
        object.__setattr__(self, 'sha256', hashlib.sha256(self.data).hexdigest())


class BundleAssembler:
    def __init__(self):
        self.buffer = bytearray(); self.body = bytearray()
        self.identity = None; self.next_chunk = 0

    def feed(self, incoming):
        self.buffer.extend(incoming)
        while len(self.buffer) >= 4:
            length, = struct.unpack_from('<I', self.buffer)
            if not 24 < length <= 65536: raise ValueError('Invalid U02H packet length')
            if len(self.buffer) < 4 + length: return
            packet = bytes(self.buffer[4:4+length]); del self.buffer[:4+length]
            magic, session, frame, total, index, count = struct.unpack_from('<4sQIIHH', packet)
            if (magic != b'U02H' or not 4 < total <= MAX_BODY
                    or count != (total+CHUNK_BYTES-1)//CHUNK_BYTES or index >= count
                    or len(packet)-24 != min(CHUNK_BYTES, total-index*CHUNK_BYTES)):
                raise ValueError('Invalid U02H fragment')
            if index == 0:
                if self.identity is not None: raise ValueError('Incomplete previous U02H bundle')
                self.identity = (session, frame, total, count); self.next_chunk = 0; self.body.clear()
            if self.identity != (session, frame, total, count) or index != self.next_chunk:
                raise ValueError('U02H fragment identity/order mismatch')
            self.body.extend(packet[24:]); self.next_chunk += 1
            if self.next_chunk == count:
                body = bytes(self.body); self.identity = None; self.body.clear()
                yield session, frame, body


def decode_bundle(body, session, source_size=(1920,1080), pair_limit_us=20000):
    if not 4 < len(body) <= MAX_BODY: raise ValueError('Invalid U02H body size')
    metadata_length, = struct.unpack_from('<I', body)
    if not 0 < metadata_length <= 16384 or 4+metadata_length >= len(body):
        raise ValueError('Invalid U02H metadata length')
    message = json.loads(body[4:4+metadata_length])
    validate_message(message, session, source_size, pair_limit_us)
    if message.get('raw_image_transport') != 'u02h_paired_jpeg/1':
        raise ValueError('Undeclared paired JPEG transport')
    descriptors = message.get('raw_images')
    if not isinstance(descriptors, list) or len(descriptors) != 2:
        raise ValueError('Two original JPEG descriptors are required')
    offset = 4+metadata_length; images = []
    for side, camera, item in zip(('left','right'), message['cameras'], descriptors):
        if any(item.get(key) != camera[key] for key in ('camera_id','frame_id','capture_timestamp_us')) or item['camera_id'] != side:
            raise ValueError('Original JPEG and FPGA frame identity mismatch')
        size = item.get('jpeg_bytes'); crc = item.get('jpeg_crc32')
        if type(size) is not int or not 0 < size <= MAX_IMAGE or offset+size > len(body):
            raise ValueError('Invalid original JPEG length')
        if type(crc) is not int or not 0 <= crc <= 0xffffffff: raise ValueError('Invalid JPEG CRC32')
        data = bytes(body[offset:offset+size]); offset += size
        if zlib.crc32(data) & 0xffffffff != crc: raise ValueError('Original JPEG CRC32 mismatch')
        if not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
            raise ValueError('Invalid JPEG boundary markers')
        images.append(data)
    if offset != len(body): raise ValueError('Unexpected bytes after paired JPEGs')
    return message, images


def save_images(output, message, images):
    folder = output/'raw_images'; folder.mkdir(exist_ok=True)
    records, payloads = [], []
    for camera, data in zip(message['cameras'], images):
        path = folder/f"{message['session_id']:016x}_{message['pair_id']}_{camera['camera_id']}.jpg"
        with path.open('xb') as stream: stream.write(data)
        record = dict(side=camera['camera_id'],path=str(path.resolve()),
            boot_id=message['boot_id'],session_id=message['session_id'],pair_id=message['pair_id'],
            frame_id=camera['frame_id'],capture_timestamp_us=camera['capture_timestamp_us'],
            source_size=list(camera['source_size']),jpeg_bytes=len(data),checksum_source='PC SHA256 after board CRC32 validation')
        payload = ReceivedJpeg(data, image_identity(record))
        record['sha256'] = payload.sha256
        records.append(record); payloads.append(payload)
    return records, tuple(payloads)


def run(self):
    connection = 0; previous = {}; intentional = False; saved_bytes = 0
    try:
        with (self.output/'received.ndjson').open('w',encoding='utf8') as log:
            while not self.stop.is_set():
                try:
                    with socket.create_connection((self.host,self.port),timeout=self.connect_timeout) as sock:
                        sock.settimeout(.02); connection += 1
                        self.events.append(dict(event='connected',connection=connection,pc_ns=time.perf_counter_ns()))
                        assembler = BundleAssembler(); clock_state = ClockProbeTracker()
                        def probe():
                            packet = clock_state.packet(time.perf_counter_ns())
                            if packet is not None: sock.sendall(packet)
                        probe()
                        while not self.stop.is_set():
                            try: incoming = sock.recv(65536)
                            except socket.timeout: continue
                            if not incoming:
                                raise ConnectionAbortedError('Peer closed the stream; waiting to reconnect, not source completion')
                            for session, frame, body in assembler.feed(incoming):
                                received = time.perf_counter_ns()
                                message, images = decode_bundle(body,session,self.expected_source_size,self.expected_pair_limit_us)
                                if (message.get('fixture_replay') or message.get('human_observed') is False) and not getattr(self,'allow_fixture',False):
                                    raise ValueError('Saved-JPEG fixture cannot enter a live human candidate')
                                old = previous.get(session)
                                if old:
                                    if message['pair_id'] <= old['pair_id']: raise ValueError('Repeated/backward FPGA pair')
                                    for cur, prior in zip(message['cameras'],old['cameras']):
                                        if (cur['capture_timestamp_us'] <= prior['capture_timestamp_us']
                                                or not 0 < ((cur['frame_id']-prior['frame_id']) & 0xffffffff) < 0x80000000):
                                            raise ValueError('Repeated/backward original camera frame')
                                clock = message['clock_probe']; latency = None
                                if clock['nonce']:
                                    clock_state.acknowledge(clock)
                                    lower = clock['board_send_ns']-received; upper = clock['board_receive_ns']-clock['pc_send_ns']
                                    if upper < lower: raise ValueError('Invalid clock exchange')
                                    latency = dict(estimated_ms=(upper-lower)/2e6,uncertainty_ms=(upper-lower)/2e6,
                                        minimum_ms=0,maximum_ms=(upper-lower)/1e6,offset_min_ns=lower,offset_max_ns=upper)
                                saved_bytes += sum(map(len,images))
                                if saved_bytes > getattr(self,'raw_image_max_bytes',512*1024*1024):
                                    raise ValueError('Paired JPEG evidence exceeds the configured recording budget')
                                try: raw_images, payloads = save_images(self.output,message,images)
                                except OSError as error: raise RuntimeError('Could not persist paired JPEG evidence') from error
                                previous[session] = message
                                row = dict(pc_receive_ns=received,connection=connection,transport_frame_id=frame,
                                    latency=latency,message=message,raw_images=raw_images,transport='u02h_paired_jpeg/1',
                                    receipt_scope='Complete measured FPGA metadata and both original JPEGs')
                                self.rows.append(row)
                                # Only the bounded compute queue owns bytes; evidence rows stay JSON-only.
                                if self.on_record is not None: self.on_record(dict(row, _verified_jpegs=payloads))
                                try:
                                    log.write(json.dumps(row,separators=(',',':'))+'\n'); log.flush(); self.rows.mark_persisted()
                                except OSError as error: raise RuntimeError('Could not persist received FPGA pair') from error
                                if self.reconnect_after and len(self.rows)>=self.reconnect_after and not intentional:
                                    intentional=True
                                    self.events.append(dict(event='intentional_disconnect',connection=connection,pc_ns=time.perf_counter_ns()))
                                    raise ConnectionAbortedError('Requested recovery test')
                                probe()
                except (OSError,ConnectionAbortedError) as error:
                    self.events.append(dict(event='disconnected',connection=connection,reason=str(error),pc_ns=time.perf_counter_ns()))
                    self.stop.wait(.2)
    except Exception:
        self.error = traceback.format_exc()
        (self.output/'receiver_error.txt').write_text(self.error,encoding='utf8')
