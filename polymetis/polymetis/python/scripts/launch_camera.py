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
        if cfg.use_depth:
            config.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.framerate)
        profile = pipeline.start(config)
        color_intrinsics = profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()
        if cfg.use_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: " , depth_scale)
            align = rs.align(rs.stream.color)
        with grpc.insecure_channel(f'{cfg.ip}:{cfg.port}') as channel:
            stub = polymetis_pb2_grpc.CameraServerStub(channel)
            stub.SendIntrinsic(
                polymetis_pb2.CameraIntrinsic(
                    coeffs=color_intrinsics.coeffs, 
                    fx=color_intrinsics.fx,
                    fy=color_intrinsics.fy,
                    height=color_intrinsics.height,
                    width=color_intrinsics.width,
                    model=color_intrinsics.model,
                    ppx=color_intrinsics.ppx,
                    ppy=color_intrinsics.ppy,
                )
            )
            if cfg.use_depth:
                stub.SendMetaData(
                    polymetis_pb2.CameraMetaData(
                        depth_scale=depth_scale
                    )
                )
            start_time = time.time()
            while True:
                frames = pipeline.wait_for_frames()
                if cfg.use_depth:
                    frames = align.process(frames)
                    # Get aligned frames
                    depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                    depth_image = np.expand_dims(np.asarray(depth_frame.get_data())[::cfg.downsample, ::cfg.downsample], axis=-1).astype(np.int16)
                color_frame = frames.get_color_frame()
                color_image = np.asarray(color_frame.get_data())[::cfg.downsample, ::cfg.downsample, :].astype(np.uint8)
                if cfg.use_depth:
                    stub.SendImage(polymetis_pb2.CameraImage(
                        width=cfg.width // cfg.downsample, height=cfg.height // cfg.downsample, channel=4, 
                        image_data=color_image.reshape(-1).tobytes(), depth_data=depth_image.reshape(-1).tobytes()
                    ))
                else:
                    # print("send shape", color_image.shape, time.time() - start_time)  # (90, 160, 3)
                    stub.SendImage(polymetis_pb2.CameraImage(
                        width=cfg.width // cfg.downsample, height=cfg.height // cfg.downsample, channel=3, 
                        image_data=color_image.reshape(-1).tobytes()
                    ))


if __name__ == "__main__":
    main()
