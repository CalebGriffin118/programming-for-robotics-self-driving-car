from vehicle import Driver
from controller import Camera, Robot, Lidar, DistanceSensor
import math
import heapq
import itertools
import numpy as np
import cv2

driver = Driver()
timestep = int(driver.getBasicTimeStep())
STATE = "DRIVING"


# get all sensors
front_bumper_camera = driver.getDevice("front_bumper_camera")
lidar = driver.getDevice("lidar")
gps = driver.getDevice("carGPS")
camera = driver.getDevice("front_camera")
compass = driver.getDevice("car_compass")
inertial_unit = driver.getDevice("car_inertial_unit")

W = None
H = None
MAX_AREA = None
POST_TURN_FRAMES = 70            # ~0.8 seconds at 32 ms timestep
MIN_CONFIRM_FRAMES = 5

if None in (front_bumper_camera, lidar, gps, camera, compass, inertial_unit):
    STATE = "ERROR"
else:
    # if all sensors are found enable them all
    lidar.enable(timestep)
    lidar.enablePointCloud()
    front_bumper_camera.enable(timestep)
    gps.enable(timestep)
    camera.enable(timestep)
    inertial_unit.enable(timestep)
    compass.enable(timestep)

    W = camera.getWidth()
    H = camera.getHeight()
    MAX_AREA = (W * H) * 0.4

rain_counter = 0
minimum_expected_acceleration = 1.0
rain_counter = 0
rain_mode = False
previous_speed = 0.0
normal_drive_steps = 0
rain_speed = 8.0
min_car_distance = 2.5
mainIntersctions = [(-61.6, -16.5, 1.4), (-38.3, -38.4, 1.4 ), (0, 0, 1.4), (-22, 22, 1.4)]
intersections = [(-33.5, -33.5, 1.4), (-43, -43, 1.4), (-43, -33, 1.4), (-4.4, -4.4, 1.4), (4.4, 4.4, 1.4), (4.4, -24.7, 1.4), (-4.9, 4.4, 1.4), (-17.5, 18, 1.4), (-26, 26, 1.4), (-26.5, 18, 1.4), (-56, -11, 1.4), (-64, -20, 1.4), (-57.8, -19, 1.4), (-65, -13, 1.4)]


alpha = 0.2
red_score = 0.0

stop_counter = 0
yield_counter = 0
traffic_counter = 0

slip_timer = 0.0
recheck_timer = 0.0

slip_detect_time = 0.2      # time allowed of bad acceleration
rain_recheck_time = 8.0
normal_speed = 20
rain_speed = 8.0

tracked_lane_x = None
previous_lane_steering = 0.0
lane_missed_frames = 0
post_turn_stabilize = 0        # NEW: frames of gentle driving after a turn


path_idx = 0
direction = None
start_heading = None
counter = 0

LANE_TARGET_X        = 265
LANE_ACCEPTABLE_LEFT = 258   # LEFT  correction finishes here
LANE_ACCEPTABLE_RIGHT= 278   # RIGHT correction finishes here
LANE_TOO_CLOSE       = 250   # LEFT  correction triggers here
LANE_TOO_FAR         = 288   # RIGHT correction triggers here
LANE_X_ALPHA         = 0.40  # line_x smoothing  (0=smooth, 1=raw)
previous_line_x = float(LANE_TARGET_X)
lane_correction_mode = "NONE"

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
        # create each main node (an intersection)
        for i in mainIntersctions:
            newMainNode = MainNode(i, True)
            self.add_children(newMainNode)
            self.intersections.append(newMainNode)

        # set the current head to the closes main node
        self.newHead(currentPos)

        # add neighbours to each main node (accessable intersections form each other)
        self.intersections[0].neighbours.append(self.intersections[1])
        self.intersections[0].neighbours.append(self.intersections[3])
        self.intersections[1].neighbours.append(self.intersections[0])
        self.intersections[1].neighbours.append(self.intersections[2])
        self.intersections[2].neighbours.append(self.intersections[1])
        self.intersections[2].neighbours.append(self.intersections[3])
        self.intersections[3].neighbours.append(self.intersections[2])
        self.intersections[3].neighbours.append(self.intersections[0])

    # add all possible turns at an intersection as a child node (SubNode)
    def add_children(self, currNode):
        for i in intersections:
            dist = self.distance(currNode.position, i)
            if 0 < dist < 15:
                currNode.children.append(SubNode(i, True))

    # calculate the distance between two points
    def distance(self, currentPos, newPos):
        pairs = zip(currentPos, newPos)
        sum = 0
        for i in pairs:
            sum += abs(i[0] - i[1])
        return sum

    # find the closest main node and set it as the current node
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


# calcualte the manhattan distance between two coordinates
def manhattan_distance(a, b):
    distance = 0
    for i in range(len(a)):
        distance += abs(a[i] - b[i])
    return distance

# use the a* heuristic to find a quick path between two nodes in the graph
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

# get the coordnate positions of a list of nodes
# @return a list of 3D coordnates
def get_coords(node_path):
    path = []
    for idx in range(len(node_path) - 1):
        path.append(node_path[idx])
        min_dist = None
        min_node = None
        # for each main node check if there is a subnode within a close distance
        for sub_node in node_path[idx].children:
            dist = manhattan_distance(node_path[idx + 1].position, sub_node.position)
            # if there is set it as a child
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_node = sub_node
        path.append(min_node)
    path.append(node_path[-1])
    return path

# convert the output of the a* algorithm (a list of node objects) to a list of coordnates with correct SubNods/turns
# @return value a list of 3D coordinates
def navigate(graph, target_pos):
    node_path = a_star(graph.currNode, target_pos)
    coord_path = get_coords(node_path)
    return coord_path

# stops the car from moving and prints the current speed
# @return void/nothing
def stopCarLIDAR(): 
    driver.setSteeringAngle(0)
    driver.setCruisingSpeed(0.0)
    driver.setBrakeIntensity(0.8)
    print(f"LIDAR OBSTACLE DETECTED!")

# turn the car depending on the direction
# @return the number of radians the car has turned
def turnCar(direction, heading_rad, start_heading):
    turned = None
    if direction == "AHEAD":
            driver.setSteeringAngle(0)
            driver.setCruisingSpeed(5)
            return turned
        
    if direction == "LEFT":
        driver.setSteeringAngle(-0.6)
        turned = (start_heading - heading_rad) % (2 * math.pi)
    else:
        driver.setSteeringAngle(0.4)
        turned = (heading_rad - start_heading) % (2 * math.pi)
    return turned



# make the car turn left or right and return its current state, the current node, and the cars heading
# @return STATE, path_idx, heading
def turning(direction, start_heading, path_idx, turn_angle=math.pi / 2):
    front_distance  = lidarDetect(lidar)
    obstacle_detected = front_distance < getDynamicMinDistance(driver.getCurrentSpeed())
    heading_rad = get_heading()

# Lidar safety override
    if obstacle_detected:
        stopCarLIDAR()
        return "TURNING", path_idx, heading_rad 
    else:
        turned = turnCar(direction, heading_rad, start_heading)
        # if we are going stright we do not need to turn anymore
        if turned is None:
            return "DRIVING", path_idx + 1, heading_rad
        
        driver.setCruisingSpeed(5)

        # If we crossed the boundary, the complement is the real angle
        if turned > math.pi:
            turned = (2 * math.pi) - turned

        turn_remaining = turn_angle - turned

        print(f"Start: {start_heading:.3f} | Current: {heading_rad:.3f} | Turned: {turned:.3f} | Remaining: {turn_remaining:.3f}")

        if turn_remaining < 0.08:
            driver.setSteeringAngle(0)
            global post_turn_stabilize
            post_turn_stabilize = POST_TURN_FRAMES
            return "DRIVING", path_idx + 1, heading_rad
        else:
            return "TURNING", path_idx, heading_rad


# get the direct (left, right, ahead) between two coordinates
# @return AHEAD or RIGHT or LEFT
def getDirection(robot_pos, target_pos):
    north = compass.getValues()
    heading_rad = math.atan2(north[0], north[2])
    heading_rad = (heading_rad + 2 * math.pi) % (2 * math.pi)

    fx = math.sin(heading_rad)
    fz = math.cos(heading_rad)

    dx = target_pos.position[0] - robot_pos.position[0]
    dz = target_pos.position[2] - robot_pos.position[2]

    cross = fx * dz - fz * dx
    print(cross)
    if abs(cross) < 0.029:
        return "AHEAD"
    elif cross > 0:
        return "RIGHT"
    else:
        return "LEFT"


# drive the car foward
# @return path_idx, STATE, heading
def drive(gps_pos, path_idx, path, STATE, drive_speed):
    dist = manhattan_distance(gps_pos, path[path_idx].position)
    print(f"Dist: {dist}")
    front_distance  = lidarDetect(lidar)
    obstacle_detected = front_distance < getDynamicMinDistance(driver.getCurrentSpeed())


# Lidar safety override
    if obstacle_detected:
        stopCarLIDAR()
    else:
        rain_active = detectRain(obstacle_detected)

        if rain_active:
            print("RAIN MODE ON")
            current_drive_speed = rain_speed
        else:
            print("RAIN MODE OFF")
            current_drive_speed = normal_speed
        
        # if we have arrived at the node move to the next one
        if isinstance(path[path_idx], MainNode) and dist < 8:
            path_idx += 1

        if path_idx >= len(path):
            return path_idx, STATE, get_heading()

        # we only turn at subnoeds so set the state to turning if we are at one
        if isinstance(path[path_idx], SubNode):
            STATE = "TURNING"
        elif isinstance(path[path_idx], MainNode):
            driver.setCruisingSpeed(drive_speed)
            STATE = "DRIVING"

        return path_idx, STATE, get_heading()

# detects distance of objects in lidar returns the closest object to the car
# @return min_dist
def lidarDetect(lidar_sensor):
    
    ranges = lidar_sensor.getRangeImage()
    if not ranges: 
        return float('inf')
    middle_ray = len(ranges)//2
    # horizontal = 512
    # middle_layer_start = 3*horizontal
    front_portion = ranges[middle_ray - 188: middle_ray + 188]
    center_rays = ranges[middle_ray - 25 : middle_ray + 25]
    main_ray = ranges[middle_ray : middle_ray]
    all_relevant = front_portion + center_rays + main_ray
    valid_distances = [d for d in all_relevant if 0<d<100]
    
    if not valid_distances:
        return float('inf')
    min_dist = min(valid_distances)
    print(f"LiDAR -> closest front obstacle: {min_dist:.2f}")
    return min_dist

# detects the amount of red pixels in the frame returns a mask of all red pixels
# @return mask
def detectRedScore(hsv):
    global red_score
 
    mask1 = cv2.inRange(hsv, (0,   70, 50), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    mask  = mask1 + mask2
 
    roi        = mask[0:int(H * 0.5), 0:int(W * 0.5)]
    detection  = np.sum(roi > 0) / roi.size
    red_score  = alpha * detection + (1 - alpha) * red_score
 
    print("RED SCORE:", red_score)
    return mask


# detects road signs returns the detected shape of a road sign
# @return detected_shape
def detectRoadSign(red_mask):
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
 
    detected_shape = None
    best_area      = 0
 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue
 
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        sides  = len(approx)
 
        if area > best_area:
            best_area = area
            if sides >= 7:
                detected_shape = "STOP"
            elif sides == 3:
                detected_shape = "GIVE_WAY"
 
    return detected_shape


# detects if there is a traffic lights and its colour
# @returns RED, GREEN, UNKNOWN
def detectTrafficLight(rgb, hsv):
    black_mask    = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))
    tl_contours, _ = cv2.findContours(
        black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
 
    for cnt in tl_contours:
        area = cv2.contourArea(cnt)
        if area < 800 or area > 8000:
            continue
 
        x, y, w, h = cv2.boundingRect(cnt)
 
        if h / (w + 1e-6) <= 1.8 or y >= H * 0.7:
            continue
 
        roi_tl  = rgb[y:y+h, x:x+w]
        hsv_tl  = cv2.cvtColor(roi_tl, cv2.COLOR_BGR2HSV)
 
        red_px   = np.sum(
            cv2.inRange(hsv_tl, (0,   70, 50), (10,  255, 255)) +
            cv2.inRange(hsv_tl, (170, 70, 50), (180, 255, 255)) > 0
        )
        green_px = np.sum(
            cv2.inRange(hsv_tl, (40, 70, 50), (90, 255, 255)) > 0
        )
 
        return "RED" if red_px > green_px else "GREEN"
 
    return "UNKNOWN"

# updates the counters for camera detection
# @return void/nothing
def updateCounters(detected_shape, traffic_state):
    global stop_counter, yield_counter, traffic_counter
 
    cap = MIN_CONFIRM_FRAMES + 2
 
    if detected_shape == "STOP"     and red_score > 0.015:
        stop_counter += 1
    else:
        stop_counter = max(0, stop_counter - 1)
 
    if detected_shape == "GIVE_WAY" and red_score > 0.015:
        yield_counter += 1
    else:
        yield_counter = max(0, yield_counter - 1)
 
    if traffic_state == "RED":
        traffic_counter += 1
    else:
        traffic_counter = max(0, traffic_counter - 1)
 
    stop_counter    = min(stop_counter,    cap)
    yield_counter   = min(yield_counter,   cap)
    traffic_counter = min(traffic_counter, cap)


# detect traffic lights and road signs
# @returns RED_LIGHT_STOP or STOP_SIGN or GIVE_WAY or DRIVE 
def cameraDetect(rgb, hsv):
    red_mask       = detectRedScore(hsv)
    detected_shape = detectRoadSign(red_mask)
    traffic_state  = detectTrafficLight(rgb, hsv)

    updateCounters(detected_shape, traffic_state)
    if traffic_counter >= MIN_CONFIRM_FRAMES and traffic_state == "RED":
        return "RED_LIGHT_STOP"
    if stop_counter  >= MIN_CONFIRM_FRAMES:
        return "STOP_SIGN"
    if yield_counter >= MIN_CONFIRM_FRAMES:
        return "GIVE_WAY"
    return "DRIVE"


# gets the mask for the white lines the camera
# @retuns numpy array
def getLaneMask():
    width  = front_bumper_camera.getWidth()
    height = front_bumper_camera.getHeight()
    image  = front_bumper_camera.getImage()
 
    img = np.frombuffer(image, np.uint8).reshape((height, width, 4)).copy()
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
 
    white_mask = cv2.inRange(
        hsv,
        np.array([0,   0, 110]),
        np.array([180, 110, 255])
    )
 
    road_mask   = np.zeros_like(white_mask)
    road_region = np.array([[
        (int(width * 0.38), int(height * 0.98)),
        (int(width * 0.99), int(height * 0.98)),
        (int(width * 0.98), int(height * 0.38)),
        (int(width * 0.58), int(height * 0.38)),
    ]], dtype=np.int32)
    cv2.fillPoly(road_mask, road_region, 255)
 
    lane_mask = cv2.bitwise_and(white_mask, road_mask)
    kernel    = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)


# gets all things from the mask that could be a line
# @return a list of all candiates 
def getLineCandidates(lane_mask):
    contours, _ = cv2.findContours(
        lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
 
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 2 or h < 4:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        candidates.append(int(moments["m10"] / moments["m00"]))
 
    print(f"Lane candidates: {candidates}")
    return candidates

# updates the correction if needed
# @retuns MOVE_LEFT or MOVE_RIGHT
def updateLaneMode(line_x):
    global lane_correction_mode, previous_lane_steering
 
    if lane_correction_mode != "NONE":
        return
 
    if line_x < LANE_TOO_CLOSE:
        lane_correction_mode   = "MOVE_LEFT"
        previous_lane_steering = 0.0
        print("Drifting towards right boundary -> Starting LEFT correction")
 
    elif line_x > LANE_TOO_FAR:
        lane_correction_mode   = "MOVE_RIGHT"
        previous_lane_steering = 0.0
        print("Drifting away from right boundary -> Starting RIGHT correction")

# calculates how much the car needs to turn left
# @return how much to steer the car left 
def computeLeftSteering(line_x, speed_factor):
    global lane_correction_mode, previous_lane_steering
 
    if line_x >= LANE_ACCEPTABLE_LEFT:
        lane_correction_mode   = "NONE"
        previous_lane_steering = 0.0
        print(f"LEFT correction complete | LineX={line_x} | Steering=0")
        return 0.0
 
    closeness_error = LANE_ACCEPTABLE_LEFT - line_x
 
    if line_x >= LANE_TOO_CLOSE:
        raw_steering      = -(0.010 + closeness_error * 0.0015) * speed_factor
        max_left_steering = -0.045
    else:
        raw_steering      = -(0.045 + closeness_error * 0.0035) * speed_factor
        max_left_steering = -0.18
 
    raw_steering = max(max_left_steering, raw_steering)
    steering     = 0.35 * previous_lane_steering + 0.65 * raw_steering
    steering     = max(max_left_steering, min(0.0, steering))
 
    previous_lane_steering = steering
    print(f"LEFT correction active | LineX={line_x} | FinishAt={LANE_ACCEPTABLE_LEFT} | Steering={steering:.3f}")
    return steering

# calculates how much the car needs to turn right
# @return how much to steer the car right 
def computeRightSteering(line_x, speed_factor):
    """Return the steering value for an active RIGHT correction."""
    global lane_correction_mode, previous_lane_steering
 
    # Safety: abort if boundary suddenly too close
    if line_x < LANE_TOO_CLOSE:
        lane_correction_mode   = "MOVE_LEFT"
        previous_lane_steering = 0.0
        print("RIGHT correction aborted -> Switching to LEFT correction")
        return -0.03
 
    if line_x <= LANE_ACCEPTABLE_RIGHT:
        lane_correction_mode   = "NONE"
        previous_lane_steering = 0.0
        print(f"RIGHT correction complete | LineX={line_x} | Steering=0")
        return 0.0
 
    distance_error = line_x - LANE_ACCEPTABLE_RIGHT
    raw_steering   = min(0.060, (0.015 + distance_error * 0.0015) * speed_factor)
    steering       = 0.35 * previous_lane_steering + 0.65 * raw_steering
    steering       = max(0.0, min(0.060, steering))
 
    previous_lane_steering = steering
    print(f"RIGHT correction active | LineX={line_x} | FinishAt={LANE_ACCEPTABLE_RIGHT} | Steering={steering:.3f}")
    return steering


# smooth the line out to reduce noise
# @return smoothed x position of the lane line
def smoothLineX(raw_line_x, previous_line_x):
    line_x = int(LANE_X_ALPHA * raw_line_x + (1 - LANE_X_ALPHA) * previous_line_x)
    previous_line_x = line_x
    return line_x

# keeps the car within its current lane
# @return how much the car needs to steer
def laneDetect():
    global previous_lane_steering, lane_correction_mode, previous_line_x
 
    # 1. Get lane mask and candidate x-positions
    lane_mask  = getLaneMask()
    candidates = getLineCandidates(lane_mask)
 
    # 2. No line visible — bleed off steering and return
    if not candidates:
        previous_lane_steering *= 0.30
        if abs(previous_lane_steering) < 0.003:
            previous_lane_steering = 0.0
        # print(f"No right boundary visible | Mode={lane_correction_mode} | Steering={previous_lane_steering:.3f}")
        return previous_lane_steering
    # 3. Smooth the raw line position
    line_x = smoothLineX(max(candidates), previous_line_x)

    # 4. Speed-dependent gain
    current_speed = abs(driver.getCurrentSpeed())
    speed_factor  = max(0.50, min(current_speed / 30.0, 1.0))
 
    # 5. Check whether a new correction should start
    updateLaneMode(line_x)
 
    # 6. Dispatch to the active correction
    if lane_correction_mode == "MOVE_LEFT":
        return computeLeftSteering(line_x, speed_factor)
 
    if lane_correction_mode == "MOVE_RIGHT":
        return computeRightSteering(line_x, speed_factor)
 
    # 7. Safe region — bleed off residual steering
    previous_lane_steering *= 0.25
    if abs(previous_lane_steering) < 0.003:
        previous_lane_steering = 0.0
 
    print(f"Lane safe | LineX={line_x} | Target={LANE_TARGET_X} | Mode={lane_correction_mode} | Steering={previous_lane_steering:.3f}")
    return previous_lane_steering

# detect if the weather condition is raining
# @return True or False
def detectRain(obstacle_detected):
    global rain_mode, previous_speed, slip_timer, recheck_timer

    actual_speed = driver.getCurrentSpeed()
    target_speed = driver.getTargetCruisingSpeed()
    delta_time = timestep / 1000.0

    acceleration = (actual_speed - previous_speed) / delta_time
    previous_speed = actual_speed

    # Do not check traction while obstacle braking is active
    if obstacle_detected:
        slip_timer = 0.0
        return rain_mode

    # If already in rain mode, stay in it for 6 seconds first
    if rain_mode:
        recheck_timer += delta_time

        if recheck_timer < rain_recheck_time:
            return True

        # After 6 seconds, turn rain mode off temporarily and test grip again
        rain_mode = False
        recheck_timer = 0.0
        slip_timer = 0.0
        print(" Rechecking traction... temporarily leaving rain mode")
        return False

    # Only detect slip when we are trying to accelerate
    trying_to_accelerate = actual_speed < target_speed - 1.5

    if trying_to_accelerate and acceleration < minimum_expected_acceleration:
        slip_timer += delta_time
        print(
            f"Possible slip | speed={actual_speed:.1f} km/h | "
            f"accel={acceleration:.2f} | slip_timer={slip_timer:.2f}s"
        )
    else:
        slip_timer = 0.0

    # If poor acceleration continues for half a second, activate rain mode
    if slip_timer >= slip_detect_time:
        rain_mode = True
        recheck_timer = 0.0
        slip_timer = 0.0
        print(" Rain / low traction detected. Switching to rain mode.")
        return True

    return False

# get the lidar detection distance based on the current speed
# @return distance for each speed band
def getDynamicMinDistance(current_speed):
    if current_speed <=6.0:
        return 1
    elif current_speed <= 20.0:
       return 3
    elif current_speed < 50.0:
       return 6
    else:
       return 2.5
    

graph = Graph(gps.getValues())
path = navigate(graph, (-22, 22, 1.4))
   

MODE = "CITY_MODE"


# main control loop
while driver.step() != -1:

    if MODE == "CITY_MODE":
        normal_speed = 20
    elif MODE == "HIGH_WAY_MODE":
        normal_speed = 40

    image = camera.getImage()

    img = np.frombuffer(image, np.uint8).reshape((H, W, 4)).copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    
    trafficDetection = cameraDetect(rgb, hsv)
    
    if trafficDetection in ["RED_LIGHT_STOP", "STOP_SIGN", "GIVE_WAY"]:
        print(trafficDetection)

    front_distance = lidarDetect(lidar)
    obstacle_detected = front_distance < getDynamicMinDistance(driver.getCurrentSpeed())

    # Lidar safety override
    if obstacle_detected:
        stopCarLIDAR()
        continue
    else:
        driver.setBrakeIntensity(0.0)
        rain_active = detectRain(obstacle_detected)

    if rain_active:
        print("Rain Mode ON")
        current_drive_speed = rain_speed
    else:
        print("Rain Mode OFF")
        current_drive_speed = normal_speed

    if path_idx >= len(path):
        STATE = "PARK"

    if STATE == "ERROR":
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(0)
        break

    elif STATE == "TURNING":
        if counter == 0:
            direction = getDirection(path[path_idx - 1], path[path_idx])
            start_heading = get_heading()
            counter = 1

        STATE, path_idx, _ = turning(direction, start_heading, path_idx)

    elif STATE == "DRIVING":
        lane_steering = laneDetect()
        driver.setSteeringAngle(lane_steering)

        # Post-turn gentle phase - gives laneDetect time to lock on again
        if post_turn_stabilize > 0:
            post_turn_stabilize -= 1
            current_drive_speed = min(current_drive_speed, 10.0)   # slow & safe while aligning
            print(f"POST-TURN STABILIZE ({post_turn_stabilize} left) | Speed capped at 10")

        path_idx, STATE, _ = drive(gps.getValues(), path_idx, path, STATE, current_drive_speed)
        counter = 0

    elif STATE == "PARK":
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(0)
        graph.newHead(gps.getValues())
        path = navigate(graph, (-38.3, -38.4, 1.4))
        STATE = "DRIVING"
        path_idx = 0
        counter = 0

    print(f"Direction: {direction}, State: {STATE}, path_idx: {path_idx}")
    print("________________________________")

