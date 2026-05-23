from vehicle import Driver
from controller import Camera, Robot
import math
import heapq
import itertools


driver = Driver()
camera = driver.getDevice("front_camera")
camera.enable(32)
compass = driver.getDevice("car_compass")
compass.enable(32)
gps = driver.getDevice("carGPS")
gps.enable(32)
mainIntersctions = [(-61.6, -16.1, 1.4), (-38.3, -38.7, 1.4), (0, 0, 1.4), (-22.2, 22.4, 1.4)]
intersections = [(-41.3, -41.3, 1.4), (-41.3, -41.3, 1.4), (-41.3, -41.3, 1.4), (-2.4, -2.3, 1.4), (2.6, 2.7, 1.4), (2.7, -2.8, 1.4), (-2.8, 2.6, 1.4), (-19.5, 19.4, 1.4), (-24.9, 24.7, 1.4), (-25, 19.8, 1.4), (-58.7, -13.8, 1.4), (-64, -19, 1.4), (-58.3, -18.6, 1.4), (-64.2, -13.7, 1.4)]
STATE = "DRIVING"

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
        # create nodes for each intersection and add possible turning points as children
        for i in mainIntersctions:
            newMainNode = MainNode(i, True)
            self.add_children(newMainNode)
            self.intersections.append(newMainNode)

        # set the current node to the closest intersection to the car
        self.newHead(currentPos)

        # add the connections between the intersections
        self.intersections[0].neighbours.append(self.intersections[1])
        self.intersections[0].neighbours.append(self.intersections[3])

        self.intersections[1].neighbours.append(self.intersections[0])
        self.intersections[1].neighbours.append(self.intersections[2])

        self.intersections[2].neighbours.append(self.intersections[1])
        self.intersections[2].neighbours.append(self.intersections[3])

        self.intersections[3].neighbours.append(self.intersections[2])
        self.intersections[3].neighbours.append(self.intersections[0])

    # add the possible turning points for each intersection
    def add_children(self, currNode):
        for i in intersections:
            dist = self.distance(currNode.position, i)
            if 0< dist <15:
                currNode.children.append(SubNode(i, True))
    
    # check the manhattan distance between two points
    def distance(self, currentPos, newPos):
        pairs = zip(currentPos, newPos)
        sum = 0
        for i in pairs:
            posCurr = i[0]
            posNew = i[1]
            sum += abs(posCurr - posNew)
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
    counter = itertools.count()  # tie-breaker for heapq
    priorityQueue = []

    # Push start node
    heapq.heappush(priorityQueue, (
        manhattan_distance(startNode.position, goalNode),  # f_score = g + h
        0,  # g_score
        next(counter),  # tie-breaker
        startNode,
        [startNode]  # path
    ))

    visited = set()

    while priorityQueue:
        f, g, _, current, path = heapq.heappop(priorityQueue)

        if current in visited:
            continue
        visited.add(current)

        # Check if goal reached
        if current.position == goalNode:
            pathPositions = []
            for node in path:
                pathPositions.append(node)
            return pathPositions

        # Only consider MainNode neighbours
        neighbors = []
        for neighbor in getattr(current, 'neighbours', []):
            neighbors.append(neighbor)

        # Explore neighbors
        for neighbor in neighbors:
            if getattr(neighbor, 'avaliable', True) and neighbor not in visited:
                newG = g + manhattan_distance(current.position, neighbor.position)
                newF = newG + manhattan_distance(neighbor.position, goalNode)
                # Build new path explicitly
                newPath = []
                for node in path:
                    newPath.append(node)
                newPath.append(neighbor)
                heapq.heappush(priorityQueue, (
                    newF,
                    newG,
                    next(counter),
                    neighbor,
                    newPath
                ))

    return None

def navigate(graph, target_pos):
    path = []
    temp = a_star(graph.currNode, target_pos)
    for idx in range(len(temp)-1):
        path.append(temp[idx])
        min_dist = None
        min_node = None
        for sub_node in temp[idx].children:
            dist = manhattan_distance(temp[idx+1].position, sub_node.position)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_node = sub_node

        path.append(min_node)
    path.append(temp[-1])

    return path

graph = Graph(gps.getValues())
path = navigate(graph, (0,0,1.4))



def calculate_steering(currPos, targetPos, max_steering_deg=40):
    north = compass.getValues()
    heading_rad = math.atan2(north[0], north[2])
    

    # i need to move 1.5708
prev_heading = None    
flipped = 0

def turning(direction, start_heading, path_idx, flipped, turn_angle=1.5708):
    """
    Turns the robot left or right by turn_angle (default 90 deg)
    direction: "LEFT" or "RIGHT"
    start_heading: compass heading when the turn started (radians)
    path_idx: current path index
    """

    if direction == "AHEAD":
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(5)
        return STATE, path_idx, flipped, heading_rad

    # Read current compass heading
    north = compass.getValues()
    heading_rad = math.atan2(north[0], north[2])
    heading_rad = (heading_rad + 2*math.pi) % (2*math.pi)

    # Compute target relative to start heading
    if direction == "LEFT":
        goal_heading = (start_heading + turn_angle) % (2*math.pi)
        driver.setSteeringAngle(0.25)
    else:
        goal_heading = (start_heading - turn_angle) % (2*math.pi)
        driver.setSteeringAngle(-0.6)

    driver.setCruisingSpeed(5)  # Must move forward for car to turn
    if prev_heading is not None and prev_heading - heading_rad > 0.5:
        flipped = 1

    if prev_heading is not None and abs(prev_heading - heading_rad) > 3:
        flipped = 2 

    print(f"Flipped: {flipped}")
    if flipped == 1:
        goal_heading -= 1.64

    if flipped == 2:
        goal_heading += 1.52
    # Signed angle difference [-pi, pi], handles wrap-around
    angle_diff = (goal_heading - heading_rad + math.pi) % (2*math.pi) - math.pi

    # Debug prints
    print(f"Start heading: {start_heading:.3f}, Current heading: {heading_rad:.3f}, Goal: {goal_heading:.3f}, Angle diff: {angle_diff:.3f}")

    # Check if the turn is complete (within tolerance)
    if abs(angle_diff) < 0.052:  # ~3° tolerance
        driver.setSteeringAngle(0)  # straighten wheels
        STATE = "DRIVING"
        path_idx += 1
    else:
        STATE = "TURNING"

    return STATE, path_idx, flipped, heading_rad

def getDirection(robot_pos, target_pos):
    """
    Returns 'LEFT', 'RIGHT', or 'AHEAD' of the target relative to robot's heading.
    Works for Webots using X/Z plane.
    """
    north = compass.getValues()
    heading_rad = math.atan2(north[0], north[2])  # atan2(East, North)
    heading_rad = (heading_rad + 2*math.pi) % (2*math.pi)

    # Forward vector in world X/Z plane
    fx = math.sin(heading_rad)
    fz = math.cos(heading_rad)

    # Vector to target in world X/Z plane
    dx = target_pos.position[0] - robot_pos.position[0]
    dz = target_pos.position[2] - robot_pos.position[2]

    # 2D cross product (forward x target vector)
    cross = fx * dz - fz * dx

    if cross > 0:
        return "LEFT"
    elif cross < 0:
        return "RIGHT"
    else:
        return "AHEAD"

def drive(gps_pos, path_idx, path, STATE):
    target_node = path[path_idx]
    dist = manhattan_distance(gps_pos, path[path_idx].position)
    print(f"Dist: {dist}")
    if isinstance(path[path_idx], MainNode) and dist < 15:
        path_idx +=1

    north = compass.getValues()
    heading = math.atan2(north[0], north[2])
    heading = (heading + 2*math.pi) % (2*math.pi)

    if path_idx >= len(path):
        return path_idx, STATE, heading
    

    # steering_angle = calculate_steering(gps_pos, path[path_idx])

    # Adjust speed: slow down near the node

    speed = 20

    if isinstance(path[path_idx], SubNode):
        STATE = "TURNING"
    elif isinstance(path[path_idx], MainNode):
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(speed)
        STATE = "DRIVING"

    return path_idx, STATE, heading

targetPos = (-41, -41, 1.4)
path_idx = 0
direction = None
heading = None
counter = 0
while driver.step() != -1:
    
    img = camera.getImage()
    rotation = compass.getValues()
    
    if path_idx >= len(path):
        STATE = "PARK"

    if counter == 0 and STATE == 'TURNING':
        direction = getDirection(path[path_idx-1], path[path_idx])
        counter +=1

    if STATE == 'TURNING':
        STATE, path_idx, flipped, prev_heading = turning(direction, heading, path_idx, flipped)  
    elif STATE == "DRIVING":
        path_idx, STATE, heading = drive(gps.getValues(), path_idx, path, STATE)
        counter =0
        flipped = 0
    elif STATE == "PARK":
        driver.setSteeringAngle(0)
        driver.setCruisingSpeed(0)
        graph.newHead(gps.getValues())
        path = navigate(graph, (-38.3, -38.7, 1.4))
        STATE = "DRIVING"
        path_idx = 0
        counter =0
        flipped = 0
    
    # print(f"gps pos: {gps.getValues()}")
    print(f"Direction: {direction}")
    # temp2 = []
    # for i in path:
    #     temp2.append(i.position)
    # print(f"Path: {temp2}")
    # print(f"path idx: {path_idx}")
    # print(f"path len: {len(path)}")
    # print(f"Compass: {rotation}")
    print("________________________________")






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
