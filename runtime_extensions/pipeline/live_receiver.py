"""Forward complete logged receiver records to the already prepared PC backend."""
from performance_receiver import Receiver as RawReceiver


class LiveReceiver(RawReceiver):
    def __init__(self, *args, backend, **kwargs):
        self.backend = backend
        self.relay_error = None
        self.forwarded = 0
        super().__init__(*args, on_record=self.forward, **kwargs)

    def forward(self, record):
        if self.relay_error: return
        try:
            self.backend.submit(record)
            self.forwarded += 1
        except Exception as error:
            self.relay_error = repr(error)

