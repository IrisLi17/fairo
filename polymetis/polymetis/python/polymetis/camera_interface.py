from typing import Any, Tuple
import grpc
import polymetis_pb2
import polymetis_pb2_grpc
import numpy as np
import torch
import cv2


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
        image_data = np.frombuffer(image.image_data, dtype=np.uint8).reshape((n_height, n_width, 3)).astype(np.float32)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        if n_channel == 4:
            depth_data = np.frombuffer(image.depth_data, dtype=np.int16).reshape((n_height, n_width, 1)).astype(np.float32)
            image_data = np.concatenate([image_data, depth_data], axis=-1)
            camera_metadata = self.stub.GetMetaData(EMPTY)
            depth_scale = camera_metadata.depth_scale
            image_data[..., -1] *= depth_scale
        return (image_data, timestamp)
    
    def get_intrinsic(self):
        camera_intrinsic: polymetis_pb2.CameraIntrinsic = self.stub.GetIntrinsic(EMPTY)
        return dict(
            model=camera_intrinsic.model,
            coeffs=np.array(camera_intrinsic.coeffs), 
            fx=camera_intrinsic.fx, 
            fy=camera_intrinsic.fy, 
            ppx=camera_intrinsic.ppx,
            ppy=camera_intrinsic.ppy
        )
    
    def deproject_pixel_to_point(self, pixel: torch.Tensor, depth):
        camera_intrinsic: polymetis_pb2.CameraIntrinsic = self.stub.GetIntrinsic(EMPTY)
        x = (pixel[0] - camera_intrinsic.ppx) / camera_intrinsic.fx
        y = (pixel[1] - camera_intrinsic.ppy) / camera_intrinsic.fy
        assert camera_intrinsic.model == 2
        r2  = x * x + y * y
        f = 1 + camera_intrinsic.coeffs[0] * r2 + camera_intrinsic.coeffs[1] * r2 * r2 + camera_intrinsic.coeffs[4] * r2 * r2 * r2
        ux = x * f + 2 * camera_intrinsic.coeffs[2] * x * y + camera_intrinsic.coeffs[3] * (r2 + 2 * x * x)
        uy = y * f + 2 * camera_intrinsic.coeffs[3] * x * y + camera_intrinsic.coeffs[2] * (r2 + 2 * y * y)
        x = ux
        y = uy
        return torch.Tensor([depth * x, depth * y, depth], device=pixel.device)
