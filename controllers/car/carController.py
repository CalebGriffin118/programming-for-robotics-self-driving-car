from vehicle import Driver
from controller import Camera

driver = Driver()

camera = driver.getDevice("front_camera")
camera.enable(32)

while driver.step() != -1:
    img = camera.getImage()
    print(camera.getWidth())

    driver.setCruisingSpeed(20)
    driver.setSteeringAngle(0.0)