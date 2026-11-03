"""Original parser, with one explicit successful-log-flush confirmation."""
import json, socket, struct, time, traceback
from performance_receiver import ClockProbeTracker, validate_message

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
                                self.rows.append(row)
                                if self.on_record is not None: self.on_record(row)
                                log.write(json.dumps(row, separators=(',', ':'))+'\n'); log.flush()
                                self.rows.mark_persisted()
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


