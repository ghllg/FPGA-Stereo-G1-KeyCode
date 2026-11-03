"""Skip empty-port timeouts only before this candidate has launched services."""
import psutil
from .candidate import G1


def listeners():
    # Inventory errors propagate: inability to inspect is not proof of vacancy.
    return [row for row in psutil.net_connections(kind='tcp')
            if row.status==psutil.CONN_LISTEN]


class StartupHealth:
    def __init__(self,owned,original):
        self.owned=owned;self.original=original
        busy=sorted({row.laddr.port for row in listeners() if row.laddr.port in (8770,8771)})
        if busy:raise RuntimeError('G1 服务端口已占用，保留原服务：'+', '.join(map(str,busy)))

    def __call__(self,port):
        current=[row for row in listeners() if row.laddr.port==port]
        if not current and not self.owned.processes:return None
        known={pid for pid,created in self.owned.processes}
        if any(row.pid not in known for row in current):
            raise RuntimeError(f'G1 端口 {port} 的监听进程不属于本次任务；保留原服务')
        # Preserve the original one-second request timeout, retry loops and
        # service/DDS readiness checks after any owned process has started.
        value=self.original(port)
        if value and (not isinstance(value,dict) or value.get('portable_root')!=str(G1)):
            raise RuntimeError(f'G1 端口 {port} 的服务身份不符；保留原服务')
        return value
