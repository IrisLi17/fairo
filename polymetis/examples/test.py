from polymetis import CameraInterface
import matplotlib.pyplot as plt


if __name__ == "__main__":
    camera = CameraInterface(ip_address="101.6.103.171")
    fig, ax = plt.subplots(1, 1)
    while True:
        (image, timestamp) = camera.read_once()
        ax.cla()
        ax.imshow(image)
        plt.pause(0.1)
        print(timestamp)