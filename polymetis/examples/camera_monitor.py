from polymetis import CameraInterface
import cv2
import numpy as np
import sys


if __name__ == "__main__":
    ip = sys.argv[1]
    camera = CameraInterface(ip_address=ip)
    (image, timestamp) = camera.read_once()
    n_channel = image.shape[-1]
    # if n_channel == 3:
    #     fig, ax = plt.subplots(1, 1)
    # else:
    #     fig, ax = plt.subplots(1, 2)
    while True:
        (image, timestamp) = camera.read_once()
        cv2.imshow("rgb", cv2.cvtColor(image[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        if n_channel == 4:
            cv2.imshow("depth", image[..., -1])
            # ax[0].cla()
            # ax[0].imshow(image[..., :3].astype(np.uint8))
            # ax[1].cla()
            # ax[1].imshow(image[..., -1])
            # plt.pause(0.1)
            # print(timestamp)
        cv2.waitKey(25)
        