# joystick
import pygame


##################################################################################

class Joy:

    def __init__(self):

        # initialize the joystick
        pygame.init()
        pygame.joystick.init()

        # check if a joystick is connected
        self.isConnected = False
        if pygame.joystick.get_count() == 0:
            
            # print warning
            print("No joystick connected.")

        # found a joystick
        else:

            # set flag
            self.isConnected = True
            
            # get the first joystick
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

            # print info
            print("Joystick connected: [{}]".format(self.joystick.get_name()))

        # joystick parameters
        self.deadzone = 0.05

        # axes values
        self.LS_X = 0.0
        self.LS_Y = 0.0
        self.RS_X = 0.0
        self.RS_Y = 0.0
        self.RT = 0.0
        self.LT = 0.0

        # button values
        self.A = False
        self.B = False
        self.X = False
        self.Y = False
        self.LB = False
        self.RB = False

        # D-Pad values (another axis)
        self.D_X = False
        self.D_Y = False

    # update the joystick inputs
    def update(self):

        # no joystick connected
        if self.joystick is None:

            print("No joystick connected. Cannot update inputs.")

        # joystick connected
        else:
            
            # process events
            pygame.event.pump()

            # read the axes
            self.LS_X = self.joystick.get_axis(0)   # right is (+)
            self.LS_Y = -self.joystick.get_axis(1)  # up is (+)
            self.RS_X = self.joystick.get_axis(3)   # right is (+)
            self.RS_Y = -self.joystick.get_axis(4)  # up is (+)
            self.LT = (self.joystick.get_axis(2) + 1.0) / 2.0  # unpressed is 0.0, fully pressed is +1.0
            self.RT = (self.joystick.get_axis(5) + 1.0) / 2.0  # unpressed is 0.0, fully pressed is +1.0

            # read the buttons
            self.A = self.joystick.get_button(0)
            self.B = self.joystick.get_button(1)
            self.X = self.joystick.get_button(2)
            self.Y = self.joystick.get_button(3)
            self.LB = self.joystick.get_button(4)
            self.RB = self.joystick.get_button(5)

            # read the D-Pad
            hat = self.joystick.get_hat(0)
            self.D_X = hat[0]  # -1 is left, +1 is right
            self.D_Y = hat[1]  # -1 is down, +1 is up

            # apply deadzone
            if abs(self.LS_X) < self.deadzone:
                self.LS_X = 0.0
            if abs(self.LS_Y) < self.deadzone:
                self.LS_Y = 0.0
            if abs(self.RS_X) < self.deadzone:
                self.RS_X = 0.0
            if abs(self.RS_Y) < self.deadzone:
                self.RS_Y = 0.0

##################################################################################


if __name__ == "__main__":

    joystick = Joy()

    while True:

        joystick.update()

        pygame.time.wait(10)