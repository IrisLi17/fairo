import pygame
import time

def main():
    pygame.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    count = 0
    while True:
        events = pygame.event.get([pygame.JOYAXISMOTION, pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION])
        for event in events:
            if event.type == pygame.JOYAXISMOTION:
                # axis 0: left joystick, + right, - left
                # axis 1: left joystick, + down, - up
                # axis 2: LT, - release, + press
                # axis 3: right joystick, + right, - left
                # axis 4: right joystick, + down, - up
                # axis 5: RT, - release, + press
                if abs(event.value) > 0.5:
                    print(count, "axismotion", event)
            elif event.type == pygame.JOYBUTTONDOWN:
                # A: button 0
                # B: button 1
                # X: button 2
                # Y: button 3
                # LB: button 4
                # RB: button 5
                print(count, "bottondown", event)
            elif event.type == pygame.JOYHATMOTION:
                # up: (0,1)
                # down: (0, -1)
                # left: (-1, 0)
                # right: (1, 0)
                print(count, "hatmotion", event)

if __name__ == "__main__":
    main()
 