"""One-shot saved-wire EOF: leave real-time reconnect behavior untouched."""
import ast,hashlib,inspect,json,textwrap,time
from pathlib import Path
from live_receiver import LiveReceiver
from performance_receiver import Receiver


def identity(message):
    return [message['session_id'],message['pair_id'],
        [c['capture_timestamp_us'] for c in message['cameras']]]


def receiver_for_fixture(source):
    identities=[]
    with (Path(source)/'pc_receiver/received.ndjson').open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():identities.append(identity(json.loads(line)['message']))
    if not identities:raise ValueError('Offline replay fixture must contain complete packets')
    return receiver_class(identities)


def receiver_class(expected_identities):
    expected_identities=json.loads(json.dumps(expected_identities))
    if not expected_identities:raise ValueError('Offline replay identities must not be empty')
    expected_count=len(expected_identities)
    expected_hash=hashlib.sha256(json.dumps(expected_identities,separators=(',',':')).encode()).hexdigest()
    # Keep the actual receiver/parser/validation/exception behavior. The sole
    # injected branch runs only at orderly socket EOF, before reconnecting.
    source=textwrap.dedent(inspect.getsource(Receiver.run))
    tree=ast.parse(source)
    candidates=[n for n in ast.walk(tree) if isinstance(n,ast.If)
        and isinstance(n.test,ast.UnaryOp) and isinstance(n.test.op,ast.Not)
        and isinstance(n.test.operand,ast.Name) and n.test.operand.id=='incoming'
        and len(n.body)==1 and isinstance(n.body[0],ast.Break)]
    if len(candidates)!=1:raise RuntimeError('Expected one unmodified orderly EOF branch')
    candidates[0].body[:0]=ast.parse('if self._offline_complete_eof(buffer,identity):\n self.stop.set()').body
    scope=dict(Receiver.run.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'isolated_saved_wire_eof','exec'),scope)

    class OfflineEOFReceiver(LiveReceiver):
        run=scope['run']

        def __init__(self,*args,**kwargs):
            self.offline_eof_verified=False
            super().__init__(*args,**kwargs)

        def finish(self):
            result=super().finish()
            if not self.offline_eof_verified:
                raise RuntimeError('Saved-wire source did not reach a verified complete orderly EOF')
            return result

        def _offline_complete_eof(self,parser_buffer,pending_identity):
            complete=(len(self.rows)==expected_count and self.forwarded==expected_count
                and not self.relay_error and not parser_buffer and pending_identity is None
                and [identity(r['message']) for r in self.rows]==expected_identities)
            if complete:
                actual_hash=hashlib.sha256(json.dumps([identity(r['message']) for r in self.rows],separators=(',',':')).encode()).hexdigest()
                record=dict(mode='verified_saved_wire_orderly_eof',eof_received_ns=time.perf_counter_ns(),
                    expected_packets=expected_count,received_packets=len(self.rows),forwarded_packets=self.forwarded,
                    expected_last_identity=expected_identities[-1],actual_last_identity=identity(self.rows[-1]['message']),
                    expected_identity_sequence_sha256=expected_hash,actual_identity_sequence_sha256=actual_hash,
                    ordered_identity_sequence_matches=True,parser_buffer_bytes=len(parser_buffer),pending_message_identity=pending_identity,
                    original_receiver_run_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    producer_ended_published=False,worker_drain='Original Backend.finish waits for OnlineSession.finish; no EOF publication here')
                (self.output/'offline_eof.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
                self.offline_eof_verified=True
            return complete
    return OfflineEOFReceiver
