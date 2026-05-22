from vehicle import Driver
from controller import Camera

driver = Driver()

camera = driver.getDevice("front_camera")
camera.enable(32)

while driver.step() != -1:
    img = camera.getImage()
    print(camera.getWidth())

    driver.setCruisingSpeed(20)
    driver.setSteeringAngle(-50)

def navigate(map, target_pos):
    # create a route to a location
    pass

def drive(gps_pos, list_of_targests, max_speed):
    # move the car to the next location
    # order of precedence of photo resistor, lidar, camera
    pass

def cameraDetect(camera_input):
    # detect objects in the camera such as signs and traffic lights
    pass

def lidarDetect(lidar_input):
    # detects distance of objects in lidar
    pass

def laneDetect():
    # postion its self within the lanes
    pass

# advance features
def detectNight():
    # use camera to check the brightness
    pass

def detectRain():
    # use traction to detect if it is raining
    pass
