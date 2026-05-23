from vehicle import Driver
from controller import Camera, Robot, Lidar, DistanceSensor
import math
import heapq
import itertools


driver = Driver()
timestep = int(driver.getBasicTimeStep())
camera = driver.getDevice("front_camera")
camera.enable(timestep)
compass = driver.getDevice("car_compass")
compass.enable(timestep)
gps = driver.getDevice("carGPS")
gps.enable(timestep)
rain_counter = 0
minimum_expected_acceleration = 1.0
rain_counter = 0
rain_mode = False
previous_speed = 0.0
normal_drive_steps = 0
normal_speed = 50.0
rain_speed = 8.0
lidar = driver.getDevice("lidar")
inertial_unit = driver.getDevice("car_inertial_unit")
inertial_unit.enable(timestep)
min_car_distance = 2.5
mainIntersctions = [(-61.6, -16.1, 1.4), (-38.3, -38.7, 1.4), (0, 0, 1.4), (-22.2, 22.4, 1.4)]
intersections = [(-41.3, -41.3, 1.4), (-41.3, -41.3, 1.4), (-41.3, -41.3, 1.4), (-2.4, -2.3, 1.4), (2.6, 2.7, 1.4), (2.7, -2.8, 1.4), (-2.8, 2.6, 1.4), (-19.5, 19.4, 1.4), (-24.9, 24.7, 1.4), (-25, 19.8, 1.4), (-58.7, -13.8, 1.4), (-64, -19, 1.4), (-58.3, -18.6, 1.4), (-64.2, -13.7, 1.4)]
STATE = "DRIVING"

if lidar is None:
    print("Lidar sensor not found")
else:
    lidar.enable(timestep)
    lidar.enablePointCloud()
    print("LiDar Enabled :)")   

def get_heading():
    rpy = inertial_unit.getRollPitchYaw()
    yaw = rpy[2]  # yaw is index 2
    return (yaw + 2 * math.pi) % (2 * math.pi)

class MainNode():
    def __init__(self, pos, avaliable):
        self.position = pos
        self.neighbours = []
        self.children = []
        self.avaliable = avaliable

class SubNode():
    def __init__(self, pos, avaliable):
        self.position = pos
        self.avaliable = avaliable

class Graph():
    def __init__(self, currentPos):
        self.intersections = []
        self.currNode = None
        for i in mainIntersctions:
            newMainNode = MainNode(i, True)
            self.add_children(newMainNode)
            self.intersections.append(newMainNode)

        self.newHead(currentPos)

        self.intersections[0].neighbours.append(self.intersections[1])
        self.intersections[0].neighbours.append(self.intersections[3])
        self.intersections[1].neighbours.append(self.intersections[0])
        self.intersections[1].neighbours.append(self.intersections[2])
        self.intersections[2].neighbours.append(self.intersections[1])
        self.intersections[2].neighbours.append(self.intersections[3])
        self.intersections[3].neighbours.append(self.intersections[2])
        self.intersections[3].neighbours.append(self.intersections[0])

    def add_children(self, currNode):
        for i in intersections:
            dist = self.distance(currNode.position, i)
            if 0 < dist < 15:
                currNode.children.append(SubNode(i, True))

    def distance(self, currentPos, newPos):
        pairs = zip(currentPos, newPos)
        sum = 0
        for i in pairs:
            sum += abs(i[0] - i[1])
        return sum

    def newHead(self, currPos):
        smallesDist = None
        for i in self.intersections:
            dist = self.distance(i.position, currPos)
            if self.currNode is None or smallesDist is None:
                smallesDist = dist
                self.currNode = i
            elif dist < smallesDist:
                smallesDist = dist
                self.currNode = i


def manhattan_distance(a, b):
    distance = 0
    for i in range(len(a)):
        distance += abs(a[i] - b[i])
    return distance


def a_star(startNode, goalNode):
    counter = itertools.count()
    priorityQueue = []
    heapq.heappush(priorityQueue, (
        manhattan_distance(startNode.position, goalNode),
        0,
        next(counter),
        startNode,
        [startNode]
    ))
    visited = set()

    while priorityQueue:
        f, g, _, current, path = heapq.heappop(priorityQueue)
        if current in visited:
            continue
        visited.add(current)

        if current.position == goalNode:
            return [node for node in path]

        for neighbor in getattr(current, 'neighbours', []):
            if getattr(neighbor, 'avaliable', True) and neighbor not in visited:
                newG = g + manhattan_distance(current.position, neighbor.position)
                newF = newG + manhattan_distance(neighbor.position, goalNode)
                newPath = list(path) + [neighbor]
                heapq.heappush(priorityQueue, (newF, newG, next(counter), neighbor, newPath))

    return None


def navigate(graph, target_pos):
    path = []
    temp = a_star(graph.currNode, target_pos)
    for idx in range(len(temp) - 1):
        path.append(temp[idx])
        min_dist = None
        min_node = None
        for sub_node in temp[idx].children:
            dist = manhattan_distance(temp[idx + 1].position, sub_node.position)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_node = sub_node
        path.append(min_node)
    path.append(temp[-1])
    return path


graph = Graph(gps.getValues())
path = navigate(graph, (0, 0, 1.4))


def turning(direction, start_heading, path_idx, turn_angle=math.pi / 2):
    front_distance  = lidarDetect(lidar)
    obstacle_detected = front_distance < min_car_distance

# Lidar safety override
    if obstacle_detected:
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(0.8)
        print(f"🚨 OBSTACLE DETECTED! Speed = {driver.getCurrentSpeed():.1f} km/h")
    else:
        if direction == "AHEAD":
            driver.setSteeringAngle(0)
            driver.setCruisingSpeed(5)
            return "DRIVING", path_idx + 1, get_heading()

        heading_rad = get_heading()

        if direction == "LEFT":
            driver.setSteeringAngle(-0.6)
            turned = (start_heading - heading_rad) % (2 * math.pi)
        else:
            driver.setSteeringAngle(0.25)
            turned = (heading_rad - start_heading) % (2 * math.pi)

        driver.setCruisingSpeed(5)

        # If we crossed the boundary, the complement is the real angle
        if turned > math.pi:
            turned = (2 * math.pi) - turned

        remaining = turn_angle - turned

        print(f"Start: {start_heading:.3f} | Current: {heading_rad:.3f} | Turned: {turned:.3f} | Remaining: {remaining:.3f}")

        if remaining < 0.052:
            driver.setSteeringAngle(0)
            return "DRIVING", path_idx + 1, heading_rad
        else:
            return "TURNING", path_idx, heading_rad


def getDirection(robot_pos, target_pos):
    north = compass.getValues()
    heading_rad = math.atan2(north[0], north[2])
    heading_rad = (heading_rad + 2 * math.pi) % (2 * math.pi)

    fx = math.sin(heading_rad)
    fz = math.cos(heading_rad)

    dx = target_pos.position[0] - robot_pos.position[0]
    dz = target_pos.position[2] - robot_pos.position[2]

    cross = fx * dz - fz * dx

    if cross > 0:
        return "RIGHT"
    elif cross < 0:
        return "LEFT"
    else:
        return "AHEAD"


def drive(gps_pos, path_idx, path, STATE):
    dist = manhattan_distance(gps_pos, path[path_idx].position)
    print(f"Dist: {dist}")
    front_distance  = lidarDetect(lidar)
    obstacle_detected = front_distance < min_car_distance

# Lidar safety override
    if obstacle_detected:
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(0.8)
        print(f"🚨 OBSTACLE DETECTED! Speed = {driver.getCurrentSpeed():.1f} km/h")
    else:
        if isinstance(path[path_idx], MainNode) and dist < 15:
            path_idx += 1

        if path_idx >= len(path):
            return path_idx, STATE, get_heading()

        if isinstance(path[path_idx], SubNode):
            STATE = "TURNING"
        elif isinstance(path[path_idx], MainNode):
            driver.setSteeringAngle(0)
            driver.setCruisingSpeed(10)
            STATE = "DRIVING"

        return path_idx, STATE, get_heading()


path_idx = 0
direction = None
start_heading = None
counter = 0

def lidarDetect(lidar_sensor):
    # detects distance of objects in lidar
    #if not lidar_sensor:
        #return False
    
    ranges = lidar_sensor.getRangeImage()
    if not ranges: 
        return float('inf')
    middle_ray = len(ranges)//2
    # horizontal = 512
    # middle_layer_start = 3*horizontal
    front_portion = ranges[middle_ray - 120 : middle_ray + 120]
    center_rays = ranges[middle_ray - 25 : middle_ray + 25]
    main_ray = ranges[middle_ray : middle_ray]
    all_relevant = front_portion + center_rays + main_ray
    valid_distances = [d for d in all_relevant if 0<d<100]
    
    if not valid_distances:
        return float('inf')
    min_dist = min(valid_distances)
    print(f"LiDAR -> closest front obstacle: {min_dist:.2f}")
    return min_dist


while driver.step() != -1:
    img = camera.getImage()

    front_distance = lidarDetect(lidar)
    obstacle_detected = front_distance < min_car_distance

# Lidar safety override
    if obstacle_detected:
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(0.0)
        driver.setBrakeIntensity(0.8)
        print(f"🚨 OBSTACLE DETECTED! Speed = {driver.getCurrentSpeed():.1f} km/h")
        continue
    else:
        driver.setBrakeIntensity(0.0)

    if path_idx >= len(path):
        STATE = "PARK"

    if STATE == "TURNING":
        if counter == 0:
            direction = getDirection(path[path_idx - 1], path[path_idx])
            start_heading = get_heading()
            counter = 1

        STATE, path_idx, _ = turning(direction, start_heading, path_idx)

    elif STATE == "DRIVING":
        path_idx, STATE, _ = drive(gps.getValues(), path_idx, path, STATE)
        counter = 0

    elif STATE == "PARK":
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(0)
        graph.newHead(gps.getValues())
        path = navigate(graph, (-38.3, -38.7, 1.4))
        STATE = "DRIVING"
        path_idx = 0
        counter = 0

    print(f"Direction: {direction}, State: {STATE}, path_idx: {path_idx}")
    print("________________________________")


def cameraDetect(camera_input):
    pass


def laneDetect():
    pass

def detectNight():
    pass

def detectRain(obstacle_detected):
    global rain_counter, rain_mode, previous_speed, normal_drive_steps

    actual_speed = driver.getCurrentSpeed()
    target_speed = driver.getTargetCruisingSpeed()

    delta_time = timestep / 1000.0
    actual_acceleration = (actual_speed - previous_speed) / delta_time
    previous_speed = actual_speed

    # Once rain has been detected, stay in rain mode.
    # Do not let the reduced rain speed reset the detection.
    if rain_mode:
        return True

    # Do not analyse traction while emergency braking.
    if obstacle_detected:
        rain_counter = 0
        normal_drive_steps = 0
        return False

    # Only detect rain while we are requesting normal driving speed.
    if target_speed < normal_speed - 0.1:
        rain_counter = 0
        normal_drive_steps = 0
        return False

    normal_drive_steps += 1

    # Ignore the first few frames after commanding acceleration.
    # The car is initially stationary, so the first readings can be noisy.
    if normal_drive_steps < 15:
        return False

    trying_to_accelerate = actual_speed < target_speed - 3.0

    # The car is meant to be gaining speed, but it is barely accelerating
    # or is actually slowing down.
    poor_traction = actual_acceleration < minimum_expected_acceleration

    if trying_to_accelerate and poor_traction:
        rain_counter += 1
        print(
            f"Possible slip: speed={actual_speed:.1f} km/h, "
            f"acceleration={actual_acceleration:.2f} km/h/s, "
            f"counter={rain_counter}"
        )
    else:
        rain_counter = max(0, rain_counter - 1)

    # At 32 ms timestep, 25 frames is about 0.8 seconds.
    if rain_counter > 25:
        rain_mode = True
        print(" Rain / Low Traction Detected! Switching to rain mode.")
        return True

    return False

