"""Exercise the live receiver and relay with saved FPGA messages over local TCP."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend import Backend, ROOT
from live_receiver import LiveReceiver
from metrics import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--name', required=True)
    parser.add_argument('--lag-ms', type=int, default=380)
    parser.add_argument('--smooth-ms',type=float,default=40)
    parser.add_argument('--no-quality-guard',action='store_true')
    parser.add_argument('--output',type=Path,help='Explicit new output directory for offline validation')
    parser.add_argument('--receipt-pacing', action='store_true',
        help='Preserve the recorded PC arrival intervals, including source/network jitter')
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output or ROOT/'outputs/full_pipeline_exec_20261022'/args.name
    out.mkdir(parents=True,exist_ok=False)
    records = [json.loads(x) for x in (source/'pc_receiver/received.ndjson').read_text().splitlines()]
    binding = json.loads((source/'audit/binding.json').read_text())
    scope = dict(mode='local_wire_saved_replay',pc_receipts_reissued=True,lag_ms=args.lag_ms,
        smooth_ms=args.smooth_ms,quality_guard=not args.no_quality_guard,
        source=str(source),cameras_opened=False,new_FPGA_inference=False,
        local_TCP_and_actual_live_receiver_exercised=True,source_timestamps_preserved=True,
        recorded_receipt_intervals_preserved=args.receipt_pacing,
        synthetic_clock_exchange_not_live_network_latency=True)
    if (source/'fixture.json').exists():
        scope['fixture']=json.loads((source/'fixture.json').read_text(encoding='utf8'))
        scope['source_timestamps_preserved']=not scope['fixture'].get('timestamps_shifted_for_explicit_load_test',False)
        scope['fixture_timestamps_preserved_during_replay']=True
    (out/'scope.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
    backend = receiver = None
    server_error = []
    probes = []
    lock = threading.Lock()
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    def serve():
        try:
            with listener.accept()[0] as connection:
                def read_probes():
                    pending = bytearray()
                    try:
                        while True:
                            data = connection.recv(64)
                            if not data: return
                            pending.extend(data)
                            while len(pending)>=16:
                                magic,nonce,stamp = struct.unpack('<4sIQ',pending[:16])
                                del pending[:16]
                                if magic != b'U02C': raise ValueError('Invalid clock probe')
                                with lock: probes.append((nonce,stamp))
                    except OSError: pass
                thread = threading.Thread(target=read_probes, daemon=True)
                thread.start()
                start = time.perf_counter_ns()
                def source_ns(row):
                    return (row['pc_receive_ns'] if args.receipt_pacing else
                            row['message']['cameras'][0]['capture_timestamp_us']*1000)
                origin = source_ns(records[0])
                for index,row in enumerate(records):
                    message = copy.deepcopy(row['message'])
                    deadline = start+source_ns(row)-origin
                    time.sleep(max(0,(deadline-time.perf_counter_ns())/1e9))
                    with lock: probe = probes.pop(0) if probes else None
                    clock = message['clock_probe']
                    clock.update(nonce=probe[0] if probe else 0, pc_send_ns=probe[1] if probe else 0,
                        board_receive_ns=message['publish_monotonic_ns'], board_send_ns=message['publish_monotonic_ns'])
                    payload = json.dumps(message,separators=(',',':')).encode()
                    data = struct.pack('<I',len(payload))+payload
                    count = (len(data)+1175)//1176
                    for chunk in range(count):
                        body = struct.pack('<4sQIIHH',b'U02P',message['session_id'],index+1,len(data),chunk,count)+data[chunk*1176:(chunk+1)*1176]
                        connection.sendall(struct.pack('<I',len(body))+body)
                connection.shutdown(socket.SHUT_WR)
        except Exception as error:
            server_error.append(repr(error))
        finally:
            listener.close()
    failure=None
    try:
        backend = Backend(out,binding,args.lag_ms,args.smooth_ms,not args.no_quality_guard)
        producer = threading.Thread(target=serve,daemon=True)
        producer.start()
        receiver = LiveReceiver(out/'pc_receiver',host='127.0.0.1',port=port,
            expected_source_size=(1920,1080),expected_pair_limit_us=16667,backend=backend)
        receiver.start()
        duration=((records[-1]['pc_receive_ns']-records[0]['pc_receive_ns'])/1e9 if args.receipt_pacing else
            (records[-1]['message']['cameras'][0]['capture_timestamp_us']-records[0]['message']['cameras'][0]['capture_timestamp_us'])/1e6)
        producer.join(max(40.,duration+15.))
        if producer.is_alive(): raise TimeoutError('Saved wire producer did not stop')
        time.sleep(.1)
        receiver.finish()
        assert receiver.relay_error is None, receiver.relay_error
        assert len(receiver.rows)==receiver.forwarded==len(records)
        assert not server_error,server_error
        backend.finish()
    except Exception as error:
        failure=repr(error)
        (out/'run_error.json').write_text(json.dumps(dict(error=failure),indent=2),encoding='utf8')
    finally:
        if receiver and receiver.thread.is_alive(): receiver.finish()
        if backend: backend.close()
        listener.close()
    shutil.copy2(out/'pc_receiver/received.ndjson',out/'replayed.ndjson')
    if (out/'backend.json').exists():
        print(json.dumps(analyze(out)['headline'],ensure_ascii=False),flush=True)
    if failure:raise RuntimeError(failure)


if __name__=='__main__': main()
