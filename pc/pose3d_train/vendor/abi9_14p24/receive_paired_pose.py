"""Minimal FPGA paired COCO17 TCP receiver; no image inference or 3D work."""
import argparse
import json
import math
from pathlib import Path
import socket
import statistics
import struct
import threading
import time
import traceback


class ClockProbeTracker:
    """One outstanding request per TCP connection, including zero-echo frames."""
    def __init__(self):
        self.nonce=1
        self.pending=None

    def packet(self, now_ns):
        if self.pending is not None:return None
        self.pending=(self.nonce,now_ns)
        return struct.pack('<4sIQ',b'U02C',self.nonce,now_ns)

    def acknowledge(self, probe):
        if not probe['nonce']:return False
        assert self.pending==(probe['nonce'],probe['pc_send_ns']), 'Unexpected clock probe echo'
        self.pending=None;self.nonce+=1
        return True


def validate_message(value, session, expected_source_size=(1280,720), expected_pair_limit_us=10000):
    assert tuple(expected_source_size) in ((1280,720),(1920,1080))
    width,height=expected_source_size
    assert value['schema'] == 2 and value['message_type'] == 'paired_coco17'
    assert value['session_id'] == session and len(value['boot_id']) == 36
    assert value['whole_network_on_board'] and value['backend'] == 'pl-stream'
    if value.get('stream_candidate')=='dual_context_int32_resident':
        assert value['stream_device_abi']==7 and value['stream_packet_version']==1
        assert value['stream_contexts']==2 and value['pl_cores']==1
    elif value.get('stream_candidate')=='shared_model_int32_pipeline':
        assert value['stream_device_abi']==9 and value['stream_packet_version']==1
        assert type(value['inflight_pairs']) is int and value['inflight_pairs'] in (1,2)
        assert value['stream_contexts']==2*value['inflight_pairs'] and value['pl_cores']==1
    assert value['keypoint_order'] == 'COCO17' and not value['exposure_sync_verified']
    assert expected_pair_limit_us in (10000,16667)
    if expected_pair_limit_us==16667:
        assert tuple(expected_source_size)==(1920,1080)
        assert value['stream_candidate']=='shared_model_int32_pipeline' and value['stream_device_abi']==9
        assert value.get('pairing_policy')=='bounded_half_frame_raw' and value.get('pair_limit_us')==16667
        assert value.get('pair_status')=='software_raw_half_frame_match'
    else:
        assert value.get('pairing_policy','strict_raw_10ms')=='strict_raw_10ms'
        assert value.get('pair_limit_us',10000)==10000
    assert len(value['cameras']) == 2 and abs(value['pair_delta_us']) <= expected_pair_limit_us
    for side, camera in zip(('left', 'right'), value['cameras']):
        assert camera['camera_id'] == side and camera['source_size'] == [width, height]
        assert camera['coordinate_space'] == 'source_image_pixels' and camera['pl_calls'] == 110
        assert camera['timestamp_source'] == 'v4l2_monotonic_unverified_exposure'
        assert camera['capture_flags'] & 0xe000 == 0x2000
        assert camera['infer_end_ns'] >= camera['infer_start_ns'] >= camera['dequeue_monotonic_ns'] >= camera['receive_monotonic_ns']
        assert value['publish_monotonic_ns'] >= camera['infer_end_ns']
        assert len(camera['keypoints']) == 17
        for x, y, score, valid in camera['keypoints']:
            assert all(math.isfinite(number) for number in (x, y, score))
            assert isinstance(valid, bool)
            assert valid == (score >= .3 and 0 <= x < width and 0 <= y < height)
    assert value['pair_delta_us'] == value['cameras'][1]['capture_timestamp_us'] - value['cameras'][0]['capture_timestamp_us']


class Receiver:
    def __init__(self, output, host='192.168.50.2', port=18084, reconnect_after=0, connect_timeout=.5,
                 expected_source_size=(1280,720), expected_pair_limit_us=10000):
        self.output = Path(output)
        self.host, self.port = host, port
        self.reconnect_after = reconnect_after
        if tuple(expected_source_size) not in ((1280,720),(1920,1080)):
            raise ValueError('Expected source size must be explicitly 720p or 1080p')
        self.expected_source_size=tuple(expected_source_size)
        if expected_pair_limit_us not in (10000,16667) or (expected_pair_limit_us==16667 and self.expected_source_size!=(1920,1080)):
            raise ValueError('Half-frame admission requires explicit 1080p experimental receiver mode')
        self.expected_pair_limit_us=expected_pair_limit_us
        if not math.isfinite(connect_timeout) or not .1 <= connect_timeout <= 2:
            raise ValueError('Connect timeout must be between 0.1 and 2 seconds')
        self.connect_timeout = connect_timeout
        self.stop = threading.Event()
        self.rows, self.events, self.error = [], [], None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(4)
        assert not self.thread.is_alive(), 'Receiver did not stop'
        intervals = [(b['pc_receive_ns'] - a['pc_receive_ns'])/1e9 for a, b in zip(self.rows, self.rows[1:]) if a['connection'] == b['connection']]
        latency = [row['latency'] for row in self.rows if row.get('latency')]
        result = dict(messages=len(self.rows), events=self.events,
                      receive_pairs_per_s=len(intervals)/sum(intervals) if intervals else 0,
                      network_delay_estimate_ms_median=statistics.median(x['estimated_ms'] for x in latency) if latency else None,
                      clock_uncertainty_ms_median=statistics.median(x['uncertainty_ms'] for x in latency) if latency else None,
                      latency_scope='Publish to PC complete JSON receipt, midpoint of four-timestamp clock bounds; includes serialization/socket/assembly; not exposure latency',
                      rows_with_clock_probe=len(latency))
        (self.output/'receiver_summary.json').write_text(json.dumps(result, indent=2)+'\n')
        if self.error:
            result['error']=self.error
            (self.output/'receiver_summary.json').write_text(json.dumps(result,indent=2)+'\n')
            raise RuntimeError(self.error)
        return result

    def run(self):
        connection = 0
        previous = {}
        intentional = False
        try:
            with (self.output/'received.ndjson').open('w', encoding='utf-8') as log:
                while not self.stop.is_set():
                    try:
                        with socket.create_connection((self.host, self.port), timeout=self.connect_timeout) as sock:
                            sock.settimeout(.5)
                            connection += 1
                            self.events.append(dict(event='connected', connection=connection, pc_ns=time.perf_counter_ns()))
                            buffer = bytearray()
                            assembly = bytearray()
                            identity = None
                            next_chunk = 0
                            clock_state=ClockProbeTracker()
                            def probe():
                                sent = time.perf_counter_ns()
                                packet=clock_state.packet(sent)
                                if packet is not None:sock.sendall(packet)
                            probe()
                            while not self.stop.is_set():
                                try:
                                    incoming = sock.recv(65536)
                                except socket.timeout:
                                    continue
                                if not incoming:
                                    break
                                buffer.extend(incoming)
                                while len(buffer) >= 4:
                                    length, = struct.unpack_from('<I', buffer)
                                    assert 24 < length <= 1200, 'Invalid packet length'
                                    if len(buffer) < 4+length:
                                        break
                                    packet = bytes(buffer[4:4+length]); del buffer[:4+length]
                                    magic, session, frame, total, index, count = struct.unpack_from('<4sQIIHH', packet)
                                    assert magic == b'U02P' and 4 < total <= 16388
                                    assert count == (total+1175)//1176 and index < count
                                    assert len(packet)-24 == min(1176, total-index*1176)
                                    if index == 0:
                                        assert identity is None, 'Incomplete previous message'
                                        identity = (session, frame, total, count)
                                        next_chunk = 0; assembly.clear()
                                    assert identity == (session, frame, total, count) and index == next_chunk
                                    assembly.extend(packet[24:]); next_chunk += 1
                                    if next_chunk != count:
                                        continue
                                    received = time.perf_counter_ns()
                                    assert len(assembly) == total
                                    metadata_length, = struct.unpack_from('<I', assembly)
                                    assert metadata_length == total-4
                                    message = json.loads(assembly[4:])
                                    validate_message(message, session,self.expected_source_size,self.expected_pair_limit_us)
                                    old = previous.get(session)
                                    if old:
                                        assert message['pair_id'] > old['pair_id']
                                        for cur, prior in zip(message['cameras'], old['cameras']):
                                            assert cur['capture_timestamp_us'] > prior['capture_timestamp_us']
                                            assert 0 < ((cur['frame_id']-prior['frame_id']) & 0xffffffff) < 0x80000000
                                    previous[session] = message
                                    clock = message['clock_probe']
                                    latency = None
                                    if clock['nonce']:
                                        clock_state.acknowledge(clock)
                                        # Offset = board-PC. With nonnegative path delay it is
                                        # bounded by b3-c4 <= offset <= b2-c1, without clock equality.
                                        lower = clock['board_send_ns']-received
                                        upper = clock['board_receive_ns']-clock['pc_send_ns']
                                        assert upper >= lower, 'Invalid clock exchange'
                                        latency = dict(estimated_ms=(upper-lower)/2e6, uncertainty_ms=(upper-lower)/2e6,
                                                       minimum_ms=0, maximum_ms=(upper-lower)/1e6,
                                                       offset_min_ns=lower, offset_max_ns=upper)
                                    row = dict(pc_receive_ns=received, connection=connection, transport_frame_id=frame,
                                               latency=latency, message=message)
                                    self.rows.append(row); log.write(json.dumps(row, separators=(',', ':'))+'\n'); log.flush()
                                    identity = None
                                    if self.reconnect_after and len(self.rows) >= self.reconnect_after and not intentional:
                                        intentional = True
                                        self.events.append(dict(event='intentional_disconnect', connection=connection, pc_ns=time.perf_counter_ns()))
                                        raise ConnectionAbortedError('Requested recovery test')
                                    # nonce=0 means the pending request has not
                                    # been handled, not permission to enqueue another.
                                    probe()
                    except (OSError, ConnectionAbortedError) as error:
                        self.events.append(dict(event='disconnected', connection=connection, reason=str(error), pc_ns=time.perf_counter_ns()))
                        self.stop.wait(.2)
        except Exception as error:
            self.error = traceback.format_exc()
            (self.output/'receiver_error.txt').write_text(self.error,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.50.2'); parser.add_argument('--port', type=int, default=18084)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--reconnect-after', type=int, default=0)
    parser.add_argument('--connect-timeout', type=float, default=.5, help='TCP handshake timeout only (0.1 to 2 seconds); receive timeout remains 0.5 s')
    parser.add_argument('--source-mode',choices=('720p','1080p'),default='720p',help='Expected capture size; select 1080p explicitly for the verified 1080p handoff or matching isolated candidates; default stays 720p')
    parser.add_argument('--pair-limit-us',type=int,choices=(10000,16667),default=10000,
                        help='16667 explicitly accepts the ABI9 half-frame raw policy; never certifies exposure sync')
    args = parser.parse_args()
    receiver = Receiver(args.output, args.host, args.port, args.reconnect_after, args.connect_timeout,
                        (1920,1080) if args.source_mode=='1080p' else (1280,720),args.pair_limit_us)
    receiver.start()
    try:
        receiver.stop.wait(args.seconds)
    except KeyboardInterrupt:
        pass
    print(json.dumps(receiver.finish(), indent=2))


if __name__ == '__main__':
    main()
