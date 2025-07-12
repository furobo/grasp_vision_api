import dh_modbus_gripper
import time


class DHGrasperInterface:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.gripper = dh_modbus_gripper.dh_modbus_gripper()
        self.enable()

    def enable(self, flag=True):
        if flag:
            self.gripper.open(self.port, self.baudrate)
            self.gripper.Initialization()
            print("Send grip init...")

        # 等待初始化完成
        while self.gripper.GetInitState() != 1:
            time.sleep(0.2)

        print("Gripper initialized")

    def set_force(self, force):
        self.gripper.SetTargetForce(force)

    def set_speed(self, speed):
        self.gripper.SetTargetSpeed(speed)

    def move_to(self, position, block=True):
        self.gripper.SetTargetPosition(position)
        if block:
            while self.gripper.GetGripState() == 0:
                time.sleep(0.2)

    def open_gripper(self):
        self.move_to(1000)

    def close_gripper(self):
        self.move_to(0)

    def get_position(self):
        return self.gripper.GetCurrentPosition()

    def get_speed(self):
        return self.gripper.GetCurrentSpeed()

    def get_force(self):
        return self.gripper.GetCurrentForce()

    def get_state(self):
        return self.gripper.GetGripState()

    def close(self):
        self.gripper.close()
        
    def move(self, position, speed, torque):
        self.set_speed(speed)
        self.set_force(torque)
        self.move_to(position)

if __name__ == '__main__':
    gripper = DHGrasperInterface()
    # gripper.move(0, 50, 50)
    # gripper.move(1000, 50, 50)
    r = gripper.get_position()
    print(r)

    