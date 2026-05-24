from vehicle import Driver
from controller import Camera, Robot, Lidar, DistanceSensor
import math
import heapq
import itertools
import numpy as np
import cv2


driver = Driver()
timestep = int(driver.getBasicTimeStep())




camera = driver.getDevice("front_camera")


compass = driver.getDevice("car_compass")

gps = driver.getDevice("carGPS")
lidar = driver.getDevice("lidar")
front_bumper_camera = driver.getDevice("front_bumper_camera")
inertial_unit = driver.getDevice("car_inertial_unit")


rain_counter = 0
minimum_expected_acceleration = 1.0
rain_counter = 0
rain_mode = False
previous_speed = 0.0
normal_drive_steps = 0
rain_speed = 8.0

route_idx = 0
ROUTE = [(-19.7, 19.4,  1.4), (-42,   -42.8, 1.4)]

min_car_distance = 2.5
STATE = "DRIVING"

alpha = 0.2
red_score = 0.0

stop_counter = 0
yield_counter = 0
traffic_counter = 0
stop_frames = 0
cooldown_frames = 0

slip_timer = 0.0
recheck_timer = 0.0

slip_detect_time = 0.2      # time allowed of bad acceleration
rain_recheck_time = 8.0
normal_speed = 20
rain_speed = 8.0

tracked_lane_x = None
previous_lane_steering = 0.0
lane_missed_frames = 0
post_turn_stabilize = 0          # NEW: frames of gentle driving after a turn
POST_TURN_FRAMES = 70            # ~0.8 seconds at 32 ms timestep

MIN_CONFIRM_FRAMES = 5


lane_correction_mode = "NONE"
LANE_TARGET_X         = 265
LANE_ACCEPTABLE_LEFT  = 258
LANE_ACCEPTABLE_RIGHT = 278
LANE_TOO_CLOSE        = 250
LANE_TOO_FAR          = 288
LANE_X_ALPHA          = 0.40
previous_line_x = float(LANE_TARGET_X)

MODE = "CITY_MODE"


path_idx = 0
direction = None
start_heading = None
counter = 0

if all(s is not None for s in [camera, compass, gps, front_bumper_camera, inertial_unit, lidar]):
    camera.enable(timestep)
    W = camera.getWidth()
    H = camera.getHeight()
    MAX_AREA = (W * H) * 0.4
    compass.enable(timestep)
    gps.enable(timestep)
    front_bumper_camera.enable(timestep)
    inertial_unit.enable(timestep)
    lidar.enable(timestep)
    lidar.enablePointCloud()
else:
    STATE = "ERROR"

# get the heading of the car
#  @return heading in radians
def get_heading():
    rpy = inertial_unit.getRollPitchYaw()
    yaw = rpy[2]  # yaw is index 2
    return (yaw + 2 * math.pi) % (2 * math.pi)

# a class for each node the car can move to 
class Node():
    def __init__(self, pos, turns, connections):
        """
        pos         - (x, y, z) position
        turns       - list of ("DIRECTION", (x, y, z)) for turns at this node
        connections - list of (x, y, z) positions this node connects to (for a*)
        """
        self.position    = pos
        self.turns       = turns        # [("LEFT", pos), ("RIGHT", pos), ("AHEAD", pos)]
        self.connections = connections  # neighbour positions for a* pathfinding
        self.neighbours  = []           # filled in by Graph2.build_neighbours()
        self.available   = True

    def getDirection(self, next_pos):
        """Return the pre-defined turn direction towards next_pos."""
        for direction, pos in self.turns:
            if pos == next_pos:
                return direction
        return "AHEAD"  # fallback


class Graph2():
    def __init__(self, currentPos):
        self.nodes   = []
        self.currNode = None

        raw = [
            ((-42,   -42.8, 1.4), [("AHEAD", (-35.3, -35,   1.4)), ("LEFT",  (-43,   -35.7, 1.4))], [(-35.3, -35, 1.4), (-43, -35.7, 1.4)]),
            ((-35.3, -35,   1.4), [("AHEAD", (-42,   -42.8, 1.4)), ("RIGHT", (-43,   -35.7, 1.4))], [(-42, -42.8, 1.4), (-43, -35.7, 1.4), (-2.8, -3.5, 1.4)]),
            ((-43,   -35.7, 1.4), [("LEFT",  (-35.3, -35,   1.4)), ("RIGHT", (-42,   -42.8, 1.4))], [(-35.3, -35, 1.4), (-42, -42.8, 1.4), (-58, -19.8, 1.4)]),
            ((-58,   -13,   1.4), [("AHEAD", (-64.7, -20,   1.4)), ("LEFT",  (-58,   -19.8, 1.4)), ("RIGHT", (-65,   -13,   1.4))], [(-64.7, -20, 1.4), (-58, -19.8, 1.4), (-65, -13, 1.4), (-25, 19.3, 1.4)]),
            ((-58,   -19.8, 1.4), [("AHEAD", (-65,   -13,   1.4)), ("LEFT",  (-64.7, -20,   1.4)), ("RIGHT", (-58,   -13,   1.4))], [(-65, -13, 1.4), (-64.7, -20, 1.4), (-58, -13, 1.4), (-43, -35.7, 1.4)]),
            ((-65,   -13,   1.4), [("AHEAD", (-58,   -19.8, 1.4)), ("LEFT",  (-58,   -13,   1.4)), ("RIGHT", (-64.7, -20,   1.4))], [(-58, -19.8, 1.4), (-58, -13, 1.4), (-64.7, -20, 1.4)]),
            ((-64.7, -20,   1.4), [("AHEAD", (-58,   -13,   1.4)), ("LEFT",  (-65,   -13,   1.4)), ("RIGHT", (-58,   -19.8, 1.4))], [(-58, -13, 1.4), (-65, -13, 1.4), (-58, -19.8, 1.4)]),
            ((-25,   24.7,  1.4), [("AHEAD", (-19.7, 19.4,  1.4)), ("RIGHT", (-25,   19.3,  1.4))], [(-19.7, 19.4, 1.4), (-25, 19.3, 1.4)]),
            ((-19.7, 19.4,  1.4), [("AHEAD", (-25,   24.7,  1.4)), ("LEFT",  (-25,   19.3,  1.4))], [(-25, 24.7, 1.4), (-25, 19.3, 1.4), (-3.8, 3.5, 1.4)]),
            ((-25,   19.3,  1.4), [("LEFT",  (-25,   24.7,  1.4)), ("RIGHT", (-19.7, 19.4,  1.4))], [(-25, 24.7, 1.4), (-19.7, 19.4, 1.4), (-58, -13, 1.4)]),
            ((-3.8,  3.5,   1.4), [("AHEAD", (3.9,   -3.8,  1.4)), ("LEFT",  (3.2,   2.1,   1.4)), ("RIGHT", (-2.8,  -3.5,  1.4))], [(3.9, -3.8, 1.4), (3.2, 2.1, 1.4), (-2.8, -3.5, 1.4), (-19.7, 19.4, 1.4)]),
            ((3.2,   2.1,   1.4), [("AHEAD", (-2.8,  -3.5,  1.4)), ("LEFT",  (3.9,   -3.8,  1.4)), ("RIGHT", (-3.8,  3.5,   1.4))], [(-2.8, -3.5, 1.4), (3.9, -3.8, 1.4), (-3.8, 3.5, 1.4)]),
            ((3.9,   -3.8,  1.4), [("AHEAD", (-3.8,  3.5,   1.4)), ("LEFT",  (-2.8,  -3.5,  1.4)), ("RIGHT", (3.2,   2.1,   1.4))], [(-3.8, 3.5, 1.4), (-2.8, -3.5, 1.4), (3.2, 2.1, 1.4)]),
            ((-2.8,  -3.5,  1.4), [("AHEAD", (3.2,   2.1,   1.4)), ("LEFT",  (-3.8,  3.5,   1.4)), ("RIGHT", (3.9,   -3.8,  1.4))], [(3.2, 2.1, 1.4), (-3.8, 3.5, 1.4), (3.9, -3.8, 1.4), (-35.3, -35, 1.4)]),
        ]

        for pos, turns, connections in raw:
            self.nodes.append(Node(pos, turns, connections))

        self.build_neighbours()
        self.newHead(currentPos)

    def build_neighbours(self):
        """Link each node to its neighbour Node objects using connection positions."""
        pos_to_node = {n.position: n for n in self.nodes}
        for node in self.nodes:
            for conn_pos in node.connections:
                neighbour = pos_to_node.get(conn_pos)
                if neighbour:
                    node.neighbours.append(neighbour)

    def distance(self, a, b):
        return sum(abs(x - y) for x, y in zip(a, b))

    def newHead(self, currPos):
        self.currNode = min(self.nodes, key=lambda n: self.distance(n.position, currPos))



# a graph representing all the intersections
# class Graph2():
#     def __init__(self, currentPos):
#         self.nodes = []
#         self.currNode = None
#         # create each node
#         self.nodes.append((-42, -42.8, 1.4), [("AHEAD", (-35.3, -35, 1.4)), ("LEFT", (-43, -35.7, 1.4))], [])
#         self.nodes.append((-35.3, -35, 1.4), [("AHEAD", (-42, -42.8, 1.4)), ("RIGHT", (-43, -35.7, 1.4))], [(-2.8, -3.5, 1.4)])
#         self.nodes.append((-43, -35.7, 1.4), [("LEFT", (-35.3, -35, 1.4)), ("RIGHT", (-42, -42.8, 1.4))], [(-58, -19.8, 1.4)])
#         self.nodes.append((-58, -13, 1.4), [("AHEAD", (-64.7, -20, 1.4)), ("LEFT", (-58, -19.8, 1.4)), ((-65, -13, 1.4), "RIGHT")], [(-25, 19.3, 1.4)])
#         self.nodes.append((-58, -19.8, 1.4), [("AHEAD", (-65, -13, 1.4)), ("LEFT", (-64.7, -20, 1.4)), ("RIGHT", (-58, -13, 1.4))], [(-43, -35.7, 1.4)])
#         self.nodes.append((-65, -13, 1.4), [("AHEAD",(-58, -19.8, 1.4)), ("LEFT",(-58, -13, 1.4)), ("RIGHT", (-64.7, -20, 1.4))], [])
#         self.nodes.append((-64.7, -20, 1.4), [("AHEAD", (-58, -13, 1.4)), ("LEFT",(-65, -13, 1.4)), ("RIGHT", (-58, -19.8, 1.4))], [])
#         self.nodes.append((-25, 24.7, 1.4), [("AHEAD",(-19.7, 19.4, 1.4)), ("RIGHT", (-25, 19.3, 1.4))], [])
#         self.nodes.append((-19.7, 19.4, 1.4), [("AHEAD",(-25, 24.7, 1.4)), ("LEFT", (-25, 19.3, 1.4))], [(-3.8, 3.5, 1.4)])
#         self.nodes.append((-25, 19.3, 1.4), [("LEFT",(-25, 24.7, 1.4)), ("RIGHT", (-19.7, 19.4, 1.4))], [(-58, -13, 1.4)])
#         self.nodes.append((-3.8, 3.5, 1.4), [("AHEAD",(3.9, -3.8, 1.4)), ("LEFT",(3.2, 2.1, 1.4)), ("RIGHT", (-2.8, -3.5, 1.4))], [(-19.7, 19.4, 1.4)])
#         self.nodes.append((3.2, 2.1, 1.4), [("AHEAD", (-2.8, -3.5, 1.4)) ("LEFT",(3.9, -3.8, 1.4)), ("RIGHT", (-3.8, 3.5, 1.4))], [])
#         self.nodes.append((3.9, -3.8, 1.4), [("AHEAD",(-3.8, 3.5, 1.4)), ("LEFT",(-2.8, -3.5, 1.4)), ("RIGHT", (3.2, 2.1, 1.4))], [])
#         self.nodes.append((-2.8, -3.5, 1.4), [("AHEAD",(3.2, 2.1, 1.4)), ("LEFT",(-3.8, 3.5, 1.4)), ("RIGHT", (3.9, -3.8, 1.4))], [(-35.3, -35, 1.4)])


#         self.newHead(currentPos)

#     # find the closest main node and set it as the current node
#     def newHead(self, currPos):
#         smallesDist = None
#         for i in self.intersections:
#             dist = self.distance(i.position, currPos)
#             if self.currNode is None or smallesDist is None:
#                 smallesDist = dist
#                 self.currNode = i
#             elif dist < smallesDist:
#                 smallesDist = dist
#                 self.currNode = i


# calcualte the manhattan distance between two coordinates
# @return distnace
def manhattan_distance(a, b):
    distance = 0
    for i in range(len(a)):
        distance += abs(a[i] - b[i])
    return distance

# use the a* heuristic to find a quick path between two nodes in the graph
# @returns list of Node objects
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
def navigate(graph, target_pos):
    path = a_star(graph.currNode, target_pos)
    if path is None:
        print(f"ERROR: No path found to {target_pos}")
        return []
    print([n.position for n in path])
    return path


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

# stops the car from moving and prints the current speed
# @return void/nothing
def stopCarLIDAR(): 
    driver.setSteeringAngle(0)
    driver.setCruisingSpeed(0.0)
    driver.setBrakeIntensity(0.8)
    # print(f"LIDAR OBSTACLE DETECTED!")

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
            return "DRIVING", path_idx, heading_rad
        
        driver.setCruisingSpeed(5)

        # If we crossed the boundary, the complement is the real angle
        if turned > math.pi:
            turned = (2 * math.pi) - turned

        turn_remaining = turn_angle - turned

        # print(f"Start: {start_heading:.3f} | Current: {heading_rad:.3f} | Turned: {turned:.3f} | Remaining: {turn_remaining:.3f}")

        if turn_remaining < 0.08:
            driver.setSteeringAngle(0)
            global post_turn_stabilize
            post_turn_stabilize = POST_TURN_FRAMES
            return "DRIVING", path_idx + 1, heading_rad
        else:
            return "TURNING", path_idx, heading_rad

# drive the car foward
# @return path_idx, STATE, heading
def drive(gps_pos, path_idx, path, STATE, drive_speed):
    dist = manhattan_distance(gps_pos, path[path_idx].position)

    front_distance = lidarDetect(lidar)
    obstacle_detected = front_distance < getDynamicMinDistance(driver.getCurrentSpeed())

    if obstacle_detected:
        stopCarLIDAR()
        return path_idx, STATE, get_heading()

    rain_mult = 0.7 if detectRain(obstacle_detected) else 1.0

    if dist < 8:
        path_idx += 1

    if path_idx >= len(path):
        return path_idx, "ARRIVED", get_heading()

    if path[path_idx].turns:
        STATE = "TURNING"
    else:
        driver.setCruisingSpeed(drive_speed * rain_mult)
        STATE = "DRIVING"

    return path_idx, STATE, get_heading()



# detects distance of objects in lidar returns the closest object to the car
# @return min_dist
def lidarDetect(lidar_sensor):
    
    ranges = lidar_sensor.getRangeImage()
    if not ranges: 
        return float('inf')
    middle_ray = len(ranges)//2
    front_portion = ranges[middle_ray - 188: middle_ray + 188]
    center_rays = ranges[middle_ray - 25 : middle_ray + 25]
    main_ray = ranges[middle_ray : middle_ray]
    all_relevant = front_portion + center_rays + main_ray
    valid_distances = [d for d in all_relevant if 0<d<100]
    
    if not valid_distances:
        return float('inf')
    min_dist = min(valid_distances)
    # print(f"LiDAR -> closest front obstacle: {min_dist:.2f}")
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
 
    # print("RED SCORE:", red_score)
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
 
        red_mask = (
            cv2.inRange(hsv_tl, (0, 70, 50), (10, 255, 255)) +
            cv2.inRange(hsv_tl, (170, 70, 50), (180, 255, 255))
        )

        red_px = np.sum(red_mask > 0)

        cv2.imshow("Red Mask", red_mask)
        cv2.waitKey(1)

        green_px = np.sum(
            cv2.inRange(hsv_tl, (40, 70, 50), (90, 255, 255)) > 0
        )
 
        return "RED" if red_px > green_px else "GREEN"
 
    return "UNKNOWN"
    
#logic for different traffic signs/lights
#@returns STOP/GO state
def handleTraffic(trafficDetection, red_score):
    global stop_frames, cooldown_frames

    if stop_frames > 0:
        stop_frames -= 1

    if cooldown_frames > 0:
        cooldown_frames -= 1

    if (trafficDetection == "STOP_SIGN" and cooldown_frames == 0 and stop_frames == 0):
        stop_frames = 94
        cooldown_frames = 156

    if stop_frames > 0:
        return "STOP"

    if trafficDetection == "RED_LIGHT_STOP" and red_score > 0.001:
        return "STOP"

    if trafficDetection == "GIVE_WAY" and cooldown_frames == 0:
        stop_frames = 30
        cooldown_frames = 150
        return "STOP"

    return "GO"
# updates the counters for camera detection
# @return void/nothing
def updateCounters(detected_shape, traffic_state):
    global stop_counter, yield_counter, traffic_counter
 
    cap = MIN_CONFIRM_FRAMES + 2
 
    if detected_shape == "STOP"     and red_score > 0.045:
        stop_counter += 1
    else:
        stop_counter = max(0, stop_counter - 1)
 
    if detected_shape == "GIVE_WAY" and red_score > 0.045:
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

    white_mask = cv2.inRange(hsv, np.array([0, 0, 110]), np.array([180, 110, 255]))

    road_region = np.array([[
        (int(width * 0.38), int(height * 0.98)),
        (int(width * 0.99), int(height * 0.98)),
        (int(width * 0.98), int(height * 0.38)),
        (int(width * 0.58), int(height * 0.38))
    ]], dtype=np.int32)

    road_mask = np.zeros_like(white_mask)
    cv2.fillPoly(road_mask, road_region, 255)

    lane_mask = cv2.bitwise_and(white_mask, road_mask)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return lane_mask

# gets all things from the mask that could be a line
# @return a list of all candiates 
def getLineCandidates(lane_mask):
    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 2 or h < 4:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        candidates.append(int(moments["m10"] / moments["m00"]))

    # print(f"Lane candidates: {candidates}")
    return candidates

# smooth the line out to reduce noise
# @return smoothed x position of the lane line
def smoothLineX(raw_x, previous_x):
    return LANE_X_ALPHA * raw_x + (1 - LANE_X_ALPHA) * previous_x

# updates the correction if needed
# @retuns MOVE_LEFT or MOVE_RIGHT
def updateLaneMode(line_x):
    global lane_correction_mode, previous_lane_steering

    if lane_correction_mode != "NONE":
        return

    if line_x < LANE_TOO_CLOSE:
        lane_correction_mode   = "MOVE_LEFT"
        previous_lane_steering = 0.0
        # print("Drifting towards right boundary -> Starting LEFT correction")

    elif line_x > LANE_TOO_FAR:
        lane_correction_mode   = "MOVE_RIGHT"
        previous_lane_steering = 0.0
        # print("Drifting away from right boundary -> Starting RIGHT correction")

# calculates how much the car needs to turn left
# @return how much to steer the car left 
def computeLeftSteering(line_x):
    global lane_correction_mode, previous_lane_steering

    if line_x >= LANE_ACCEPTABLE_LEFT:
        lane_correction_mode   = "NONE"
        previous_lane_steering = 0.0
        # print(f"LEFT correction complete | LineX={line_x} | Steering=0")
        return 0.0

    speed       = abs(driver.getCurrentSpeed())
    sf          = max(0.50, min(speed / 30.0, 1.0))
    error       = LANE_ACCEPTABLE_LEFT - line_x

    if line_x >= LANE_TOO_CLOSE:
        raw          = -(0.010 + error * 0.0015) * sf
        max_steering = -0.045
    else:
        raw          = -(0.045 + error * 0.0035) * sf
        max_steering = -0.18

    raw      = max(max_steering, raw)
    steering = max(max_steering, min(0.0, 0.25 * previous_lane_steering + 0.75 * raw))

    previous_lane_steering = steering
    # print(f"LEFT correction active | LineX={line_x} | FinishAt={LANE_ACCEPTABLE_LEFT} "
    #       f"| Speed={speed:.1f} | Steering={steering:.3f}")
    return steering

# calculates how much the car needs to turn right
# @return how much to steer the car right 
def computeRightSteering(line_x):
    global lane_correction_mode, previous_lane_steering

    if line_x < LANE_TOO_CLOSE:
        lane_correction_mode   = "MOVE_LEFT"
        previous_lane_steering = 0.0
        # print("RIGHT correction aborted -> Switching to LEFT correction")
        return -0.03

    if line_x <= LANE_ACCEPTABLE_RIGHT:
        lane_correction_mode   = "NONE"
        previous_lane_steering = 0.0
        # print(f"RIGHT correction complete | LineX={line_x} | Steering=0")
        return 0.0

    speed  = abs(driver.getCurrentSpeed())
    sf     = max(0.50, min(speed / 30.0, 1.0))
    error  = line_x - LANE_ACCEPTABLE_RIGHT
    raw    = min(0.060, (0.015 + error * 0.0015) * sf)
    steering = max(0.0, min(0.060, 0.40 * previous_lane_steering + 0.60 * raw))

    previous_lane_steering = steering
    # print(f"RIGHT correction active | LineX={line_x} | FinishAt={LANE_ACCEPTABLE_RIGHT} "
    #       f"| Speed={speed:.1f} | Steering={steering:.3f}")
    return steering

# keeps the car within its current lane
# @return how much the car needs to steer
def laneDetect():
    global previous_lane_steering, lane_correction_mode, previous_line_x

    lane_mask  = getLaneMask()
    candidates = getLineCandidates(lane_mask)

    if not candidates:
        previous_lane_steering *= 0.30
        if abs(previous_lane_steering) < 0.003:
            previous_lane_steering = 0.0
        # print(f"No right boundary visible | Mode={lane_correction_mode} | Steering={previous_lane_steering:.3f}")
        return previous_lane_steering

    line_x = smoothLineX(max(candidates), previous_line_x)
    previous_line_x = line_x

    updateLaneMode(line_x)

    if lane_correction_mode == "MOVE_LEFT":
        return computeLeftSteering(line_x)

    if lane_correction_mode == "MOVE_RIGHT":
        return computeRightSteering(line_x)

    # Safe zone
    previous_lane_steering *= 0.25
    if abs(previous_lane_steering) < 0.003:
        previous_lane_steering = 0.0
    # print(f"Lane safe | LineX={line_x:.1f} | Target={LANE_TARGET_X} | Steering={previous_lane_steering:.3f}")
    return previous_lane_steering

# detect if the weather condition is raining
# @return True or False
def detectRain(obstacle_detected):
    global rain_mode, previous_speed
    global slip_timer, recheck_timer

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
        # print("Rechecking traction... temporarily leaving rain mode")
        return False

    # Only detect slip when we are trying to accelerate
    trying_to_accelerate = actual_speed < target_speed - 1.5

    if trying_to_accelerate and acceleration < minimum_expected_acceleration:
        slip_timer += delta_time
        # print(
        #     f"Possible slip | speed={actual_speed:.1f} km/h | "
        #     f"accel={acceleration:.2f} | slip_timer={slip_timer:.2f}s"
        # )
    else:
        slip_timer = 0.0

    # If poor acceleration continues for half a second, activate rain mode
    if slip_timer >= slip_detect_time:
        rain_mode = True
        recheck_timer = 0.0
        slip_timer = 0.0
        # print("Rain / low traction detected. Switching to rain mode.")
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

graph = Graph2(gps.getValues())
path = navigate(graph, ROUTE[0])

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
    traffic_state = handleTraffic(trafficDetection, red_score)
    
    if traffic_state == "STOP":
        driver.setCruisingSpeed(0)
        driver.setBrakeIntensity(0.8)
        continue
    else:
        driver.setBrakeIntensity(0.0)

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
        # print("Rain Mode ON")
        current_drive_speed = rain_speed
    else:
        # print("Rain Mode OFF")
        current_drive_speed = normal_speed

    if path_idx >= len(path) and route_idx >= len(ROUTE):
        STATE = "PARK"

    if STATE == "ERROR":
        break
    elif STATE == "TURNING":
        if counter == 0:
            direction = path[path_idx].getDirection(path[path_idx + 1].position)
            start_heading = get_heading()
            counter = 1

        STATE, path_idx, _ = turning(direction, start_heading, path_idx)

    elif STATE == "DRIVING":
        lane_steering = laneDetect()
        driver.setSteeringAngle(lane_steering)
        path_idx, STATE, _ = drive(gps.getValues(), path_idx, path, STATE, normal_speed)
        counter = 0

    elif STATE == "PARK":
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(0)

    elif STATE == "ARRIVED":
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(0)
        route_idx += 1
        if route_idx < len(ROUTE):
            
            graph.newHead(gps.getValues())
            path = navigate(graph, ROUTE[route_idx])
            STATE = "DRIVING"
            path_idx = 0
            counter = 0
        else:
            STATE = "PARK"
    # temp = []
    # for i in path:
    #     temp.append(i.position)
    # print(temp)
    # print(f"Direction: {direction}, State: {STATE}, path_idx: {path_idx}")
    # print(f"GPS: {path_idx}")
    # print("________________________________")

