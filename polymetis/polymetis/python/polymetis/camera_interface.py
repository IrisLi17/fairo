from typing import Any, Tuple
import grpc
import polymetis_pb2
import polymetis_pb2_grpc
import numpy as np


EMPTY = polymetis_pb2.Empty()

class CameraInterface:
    def __init__(self, ip_address: str = "localhost", port: int = 50053) -> None:
        self.channel = grpc.insecure_channel(f"{ip_address}:{port}")
        self.stub = polymetis_pb2_grpc.CameraServerStub(self.channel)

    def read_once(self) -> Tuple[np.ndarray, Any]:
        stamped_image: polymetis_pb2.CameraTimeStampedImage = self.stub.GetLatestImage(EMPTY)
        image = stamped_image.image
        timestamp = stamped_image.timestamp
        n_width = image.width
        n_height = image.height
        n_channel = image.channel
        # convert bgr to rgb
        image_data = np.array(image.image_data).reshape((n_height, n_width, n_channel)).astype(np.uint8)[..., ::-1]
        return (image_data, timestamp)
    