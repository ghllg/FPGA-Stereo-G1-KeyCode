"""Synthetic network probe only; this setting is not enabled in live capture."""
import ctypes


def immediate_ack(connection):
    # Microsoft WinSock metadata: SIO_TCP_SET_ACK_FREQUENCY = 0x98000017.
    # The frequency=1 request is also used by Cygwin's Windows TCP_QUICKACK implementation.
    api = ctypes.WinDLL('Ws2_32')
    api.WSAIoctl.argtypes = [ctypes.c_size_t,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong,
        ctypes.c_void_p,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    frequency,returned = ctypes.c_uint32(1),ctypes.c_ulong()
    result = api.WSAIoctl(connection.fileno(),0x98000017,ctypes.byref(frequency),4,None,0,ctypes.byref(returned),None,None)
    if result:
        raise OSError(api.WSAGetLastError(),'Per-socket immediate ACK request failed')

