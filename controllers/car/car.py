from controller import Robot

TIMESTEP = 64
MAX_SPEED = 10


def runRobot(Robot):
    # enable motors
    wheels = []
    wheel_names = ['wheel1', 'wheel2', 'wheel3', 'wheel4'] 

    for idx in range(len(wheel_names)):
        wheels.append(robot.getDevice(wheel_names[idx]))
        wheels[idx].setPosition(float('inf'))
        wheels[idx].setVelocity(1.0)

    # enable sensors
    light_sensors = []
    light_sensor_names = ['ls_left', 'ls_right']
    for idx in range(len(light_sensor_names)):
        light_sensors.append(robot.getDevice(light_sensor_names[idx]))
        light_sensors[idx].enable(TIMESTEP)
    
    while robot.step(TIMESTEP) != -1:
        left_light_val = light_sensors[0].getValue()/100
        right_light_val = light_sensors[1].getValue()/100

        # print("____________________________")
        # print(f"Right Speed:{right_light_val}")
        # print(f"Left Speed:{left_light_val}")

        left_speed = 2
        right_speed = 1
        wheels[0].setVelocity(left_speed)
        wheels[2].setVelocity(left_speed)

        wheels[1].setVelocity(right_speed)
        wheels[3].setVelocity(right_speed)

if __name__ == "__main__":
    robot = Robot()
    runRobot(robot)

