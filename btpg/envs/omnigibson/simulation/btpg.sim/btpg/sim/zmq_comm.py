#!/usr/bin/env python3

import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import pathlib
ROOT_PATH = pathlib.Path(__file__).parents[3]
sys.path.append(ROOT_PATH / "btpg.external")
print(ROOT_PATH)
print(ROOT_PATH / "btpg.external")

import zmq
import argparse
from typing import Optional, Dict
from typing_extensions import Protocol
import logging

from threading import Lock

##############################################################################

#!/usr/bin/env python3

import hashlib
import pickle
import sys
import numpy as np
import cv2
import pickle
import zlib
import lz4.frame
from typing import Tuple, Callable


def mat_to_jpeg(img):
    """Compresses a numpy array into a JPEG byte array."""
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def jpeg_to_mat(buf):
    """Decompresses a JPEG byte array into a numpy array."""
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


def compute_hash(obj):
    pickle_str = pickle.dumps(obj)
    # Compute MD5 hash of the string
    return hashlib.md5(pickle_str).hexdigest()


def print_size(obj):
    size = sys.getsizeof(obj)
    mb_size = size / 1024 ** 2
    print(f"The size of the object is {mb_size} MB")


def print_error(text):
    # Red color
    print(f"\033[91m{text}\033[00m")


def print_warning(text):
    # Yellow color
    print(f"\033[93m{text}\033[00m")


def make_compression_method(compression: str) -> Tuple[Callable, Callable]:
    """
    NOTE: lz4 is faster than zlib, but zlib has better compression ratio
        :return: compress, decompress functions
            def compress(object) -> bytes
            def decompress(data) -> object
    TODO: support msgpack
    """
    if compression == 'lz4':
        def compress(data): return lz4.frame.compress(pickle.dumps(data))
        def decompress(data): return pickle.loads(lz4.frame.decompress(data))
    elif compression == 'zlib':
        def compress(data): return zlib.compress(pickle.dumps(data))
        def decompress(data): return pickle.loads(zlib.decompress(data))
    else:
        raise Exception(f"Unknown compression algorithm: {compression}")
    return compress, decompress




class CallbackProtocol(Protocol):
    def __call__(self, message: Dict) -> Dict:
        ...


class ReqRepServer:
    def __init__(self,
                 port=5556,
                 impl_callback: Optional[CallbackProtocol] = None,
                 log_level=logging.DEBUG,
                 compression: str = 'lz4'):
        """
        Request reply server
        """
        self.impl_callback = impl_callback
        self.compress, self.decompress = make_compression_method(compression)
        self.port = port
        self.reset()
        logging.basicConfig(level=log_level)
        logging.debug(f"Req-rep server is listening on port {port}")

    def run(self):
        if self.is_kill:
            logging.debug("Server is prev killed, reseting...")
            self.reset()
        while not self.is_kill:
            try:
                #  Wait for next request from client
                message = self.socket.recv()
                message = self.decompress(message)
                logging.debug(f"Received new request: {message}")

                #  Send reply back to client
                if self.impl_callback:
                    res = self.impl_callback(message)
                    res = self.compress(res)
                    self.socket.send(res)
                else:
                    logging.warning("No implementation callback provided.")
                    self.socket.send(b"World")
            except zmq.Again as e:
                continue
            except zmq.ZMQError as e:
                # Handle ZMQ errors gracefully
                if self.is_kill:
                    logging.debug("Stopping the ZMQ server...")
                    break
                else:
                    raise e

    def stop(self):
        self.is_kill = True
        self.socket.close()
        self.context.term()
        del self.socket
        del self.context

    def reset(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.socket.setsockopt(zmq.SNDHWM, 5)

        # Set a timeout for the recv method (e.g., 1.5 second)
        self.socket.setsockopt(zmq.RCVTIMEO, 1500)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.is_kill = False


##############################################################################

class ReqRepClient:
    def __init__(self,
                 ip: str,
                 port=5556,
                 timeout_ms=800,
                 log_level=logging.DEBUG,
                 compression: str = 'lz4'):
        """
        :param ip: IP address of the server
        :param port: Port number of the server
        :param timeout_ms: Timeout in milliseconds
        :param log_level: Logging level, defaults to DEBUG
        :param compression: Compression algorithm, defaults to lz4
        """
        self.context = zmq.Context()
        logging.basicConfig(level=log_level)
        logging.debug(f"Req-rep client is connecting to {ip}:{port}")

        self.compress, self.decompress = make_compression_method(compression)
        self.socket = None
        self.ip, self.port, self.timeout_ms = ip, port, timeout_ms
        self._internal_lock = Lock()
        self.reset_socket()

    def reset_socket(self):
        """
        Reset the socket connection, this is needed when REQ is in a
        broken state.
        """
        if self.socket:
            self.socket.close()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.ip}:{self.port}")

    def send_msg(self, request: dict, wait_for_response=True) -> Optional[str]:
        if self.socket is None or self.socket.closed:
            logging.debug("WARNING: Socket is closed, reseting...")
            return None

        serialized = self.compress(request)
        with self._internal_lock:
            try:
                self.socket.send(serialized)
                if wait_for_response is False:
                    return None
                message = self.socket.recv()
                return self.decompress(message)
            except Exception as e:
                # accepts timeout exception
                logging.warning(
                    f"Failed to send message to {self.ip}:{self.port}: {e}, potential timeout")
                logging.debug(f"WARNING: No res from server. reset socket.")
                self.reset_socket()
                return None

    def __del__(self):
        if self.socket:
            self.socket.close()
        self.context.term()


##############################################################################

if __name__ == "__main__":
    # NOTE: This is just for Testing
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    args = parser.parse_args()

    def do_something(message):
        return b'World'

    if args.server:
        ss = ReqRepServer(port=args.port, impl_callback=do_something)
        ss.run()
    elif args.client:
        sc = ReqRepClient(ip=args.ip, port=args.port)
        r = sc.send_msg({'hello': 1})
        print(r)
    else:
        raise Exception('Must specify --server or --client')
