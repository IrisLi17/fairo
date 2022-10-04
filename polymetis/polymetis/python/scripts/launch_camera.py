import os
import time
import hydra
import grpc
import numpy as np
from polymetis.robot_servers import CameraServerLauncher
from polymetis.utils.grpc_utils import check_server_exists
import polymetis_pb2
import polymetis_pb2_grpc


@hydra.main(config_name="launch_camera")
def main(cfg):
    if cfg.server_only:
        pid = os.getpid()
    else:
        pid = os.fork()
    
    if pid > 0:
        # server
        camera_server = CameraServerLauncher(cfg.ip, cfg.port)
        camera_server.run()
    
    else:
        # camera node
        t0 = time.time()
        while not check_server_exists(cfg.ip, cfg.port):
            time.sleep(1)
            if time.time() - t0 > cfg.time_out:
                raise ConnectionError("Camera node: fail to locate server")
        
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.framerate)
        pipeline.start(config)
        with grpc.insecure_channel(f'{cfg.ip}:{cfg.port}') as channel:
            stub = polymetis_pb2_grpc.CameraServerStub(channel)
            start_time = time.time()
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asarray(color_frame.get_data())[::cfg.downsample, ::cfg.downsample, :]
                # print("send shape", color_image.shape, time.time() - start_time)  # (90, 160, 3)
                color_image = np.reshape(color_image, -1)
                stub.SendImage(polymetis_pb2.CameraImage(
                    width=cfg.width // cfg.downsample, height=cfg.height // cfg.downsample, channel=3, 
                    image_data=color_image.tolist()
                ))


if __name__ == "__main__":
    main()
