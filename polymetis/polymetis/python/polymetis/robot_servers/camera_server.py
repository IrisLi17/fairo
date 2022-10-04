from collections import deque
from concurrent import futures
from google.protobuf.timestamp_pb2 import Timestamp
import grpc
import polymetis_pb2
import polymetis_pb2_grpc
import time


class CameraServicer(polymetis_pb2_grpc.CameraServerServicer):
    def __init__(self):
        super().__init__()
        self.image_buffer = deque(maxlen=10)
    
    def SendImage(self, request: polymetis_pb2.CameraImage, context):
        # now_time = time.time()
        # seconds = int(now_time)
        # nanos = int((now_time - seconds) * 10**9)
        # timestamp = Timestamp(seconds=seconds, nanos=nanos)
        # print("input timestamp", timestamp)
        timed_image = polymetis_pb2.CameraTimeStampedImage(image=request)
        timed_image.timestamp.GetCurrentTime()
        # print("input timestamp", timed_image.timestamp)
        self.image_buffer.append(timed_image)
        return polymetis_pb2.CameraStatus(ok=True)
    
    def GetLatestImage(self, request, context):
        if len(self.image_buffer) > 0:
            return self.image_buffer[-1]
        return None


class CameraServerLauncher:
    def __init__(self, ip="localhost", port="50053"):
        self.address = f"{ip}:{port}"
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        polymetis_pb2_grpc.add_CameraServerServicer_to_server(
            CameraServicer(), self.server
        )
        self.server.add_insecure_port(self.address)
    
    def run(self):
        self.server.start()
        print(f"Camera server running at {self.address}.")
        self.server.wait_for_termination()
