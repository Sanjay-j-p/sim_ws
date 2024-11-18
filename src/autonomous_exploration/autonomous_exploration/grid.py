import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import heapq
import matplotlib.pyplot as plt
from time import sleep
from visualization_msgs.msg import Marker
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Quaternion, QuaternionStamped, Pose, Twist
from nav_msgs.msg import Odometry
import tf2_ros
import tf2_geometry_msgs  # To transform geometry_msgs types
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped


# Directions for moving in the grid (right, down, left, up)
DIRECTIONS = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]  # Right, Down, Left, Up

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Initialize variables
        self.grid = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.width = None
        self.height = None
        self.data = None
        
        self.flag=True
        self.MAX_LINEAR_VELOCITY = 1.5 
        self.kp_ang = 1.9
        self.ki_ang = 0.001
        self.kd_ang = 0.001
        self.prev_orr_error = 0.0
        self.integral_orr_error = 0.0
        
        self.map_received = False
        self.odom_received = False
        self.kp_lin = 1.0
        self.ki_lin = 0.001
        self.kd_lin = 0.001
        self.prev_distance = 0.0
        self.integral_distance = 0.0

        self.time_step=0.1

        # Subscription to the 'map' topic
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.cmd_vel = Twist()

        self.marker_pub = self.create_publisher(Marker, 'robot_path', 10)
        self.marker_pub1 = self.create_publisher(Marker, 'planned_path', 10)

        self.subscription = self.create_subscription( Odometry, 'odom', self.odom_callback, 10 )
        # sleep(1)
        self.subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
    def odom_callback(self, msg):
    # The position and orientation from the 'odom' frame
        odom_position = msg.pose.pose.position
        odom_orientation = msg.pose.pose.orientation

        try:
            # Look up the transformation from 'odom' frame to 'map' frame
            transform = self.tf_buffer.lookup_transform(
                'map',  # Target frame
                'odom',  # Source frame
                rclpy.time.Time()  # Timestamp (use current time)
            )

            # Create a PoseStamped message for the pose in the 'odom' frame
            odom_pose_stamped = PoseStamped()
            odom_pose_stamped.header = msg.header  # Use the same timestamp
            odom_pose_stamped.pose.position = odom_position
            odom_pose_stamped.pose.orientation = odom_orientation

            # Transform the entire pose from the 'odom' frame to the 'map' frame
            transformed_pose = self.tf_buffer.transform(odom_pose_stamped, 'map')

            # The transformed pose is now in the 'map' frame
            self.transformed_position = transformed_pose.pose.position
            transformed_orientation = transformed_pose.pose.orientation

            # Log the transformed position
            # self.get_logger().info(f"Transformed Position: x = {transformed_position.x}, y = {transformed_position.y}, z = {transformed_position.z}")

            # Log the transformed orientation (as quaternion)
            # self.get_logger().info(f"Transformed Orientation (Quaternion): x = {transformed_orientation.x}, y = {transformed_orientation.y}, z = {transformed_orientation.z}, w = {transformed_orientation.w}")
            transformed_orientation=[transformed_orientation.x,transformed_orientation.y,transformed_orientation.z,transformed_orientation.w]
            # Optionally, convert quaternion to Euler angles (roll, pitch, yaw)
            roll, pitch, self.yaw = euler_from_quaternion(transformed_orientation)
            # self.get_logger().info(f"Yaw (Euler angle): {self.yaw}")
            self.odom_received = True
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # print()
            b=2
            # self.get_logger().error(f"Transform failed: {e}")

    def map_callback(self, msg: OccupancyGrid):
        self.map_received = True
        """Callback function that processes the incoming map and detects frontiers."""
        self.map_data = msg
        self.resolution = self.map_data.info.resolution
        self.origin_x = self.map_data.info.origin.position.x
        self.origin_y = self.map_data.info.origin.position.y
        self.width = self.map_data.info.width
        self.height = self.map_data.info.height
        self.data = self.map_data.data

        # Convert the 1D list of occupancy values into a 2D NumPy array
        self.grid = np.array(self.data).reshape((self.height, self.width))

        # Debug: print grid size and value counts
        unique, counts = np.unique(self.grid, return_counts=True)
        # print(f"Unique grid values and counts: {dict(zip(unique, counts))}")
        
        # Check for occupancy grid values
        # print(f"Grid shape: {self.grid.shape}")
        # print(f"Sample of grid data: \n{self.grid[:100, :100]}")
        

        # Detect frontiers in the map
    def run(self):
        while not self.map_received or not self.odom_received:
            print(self.odom_received,self.map_received)
            self.get_logger().info("Waiting for map and odometry data...")
            # return  # Exit if map or odometry data is not ready
        
        self.get_logger().info("Starting frontier exploration.")
        self.expand_obstacles()
        frontiers = self.detect_frontiers()

        if frontiers:
            # Pick the first frontier and plan a path to it
            target = frontiers[0]  # Pick the first frontier (simplified)
            # print(frontiers)
            target=target[::-1]
            path = self.plan_path_to_frontier(target)
            if path:
                
                self.i=0
                # self.visualize_path(path)
                print("----------------",path)
                path=self.convert_to_real_coordinates_path(path)
                self.path_ploting(path)
                print("----------------",path)
                while True:
                    self.flag=True
                    
                    current_goal = path[self.i]
                
                    
                    speed, heading = self.path_follower( current_goal)
                    
                    while self.flag==True:

                        self.flag=self.move_ttbot(speed, heading)
                    print(len(path),self.i)
                    if len(path)==self.i:
                        print("Completed")
                        break
                    # self.visualize_path(path)
            else:
                self.get_logger().info("No valid path to frontier.")
        else:
            self.get_logger().info("No frontiers found.")

    def path_ploting(self,path):
        
        for j in range(len(path)):
            current_goal = path[j]
            
            marker = self.create_marker(current_goal[0], current_goal[1], j,1.0,0.0,0.0)
            # if marker is not None:
            self.marker_pub.publish(marker)
            sleep(0.1) 

    def path_follower(self, current_goal_pose):

        vehicle_position = self.transformed_position
        vehicle_position_values = (vehicle_position.x, vehicle_position.y, vehicle_position.z)
        # vehicle_orientation = vehicle_pose.pose.orientation
        # vehicle_orientation_values = (vehicle_orientation.x, vehicle_orientation.y, vehicle_orientation.z, vehicle_orientation.w)
        speed = current_goal_pose
       
        heading = np.arctan2((current_goal_pose[1]-vehicle_position_values[1]),(current_goal_pose[0]-vehicle_position_values[0])) 
     
        
        return speed, heading

    def move_ttbot(self, speed, heading):

        
        des_heading=heading
        des_position=speed
        # vehicle_orientation=self.ttbot_pose.pose.orientation
        # vehicle_orientation_values = (vehicle_orientation.x, vehicle_orientation.y, vehicle_orientation.z, vehicle_orientation.w)
        current_heading=self.yaw

        # print(current_heading,(des_heading - current_heading + np.pi) % (2 * np.pi) - np.pi)
        orr_error = (des_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
        pos_error_x = des_position[0] - self.transformed_position.x
        pos_error_y = des_position[1] - self.transformed_position.y
        distance = np.sqrt(pos_error_x**2 + pos_error_y**2)

    

        if abs(orr_error) > 0.04: 
            orr_error_derivative = (orr_error - self.prev_orr_error) / self.time_step
            self.integral_orr_error += orr_error * self.time_step
            angular_velocity = self.kp_ang * orr_error 
            self.cmd_vel.linear.x = 0.0  
            self.cmd_vel.angular.z = angular_velocity
            self.prev_orr_error = orr_error
        else:
            self.cmd_vel.angular.z = 0.0  
             
            if distance > 0.03:
                
                distance_derivative = (distance - self.prev_distance) / self.time_step
                self.integral_distance += distance * self.time_step
                linear_velocity = 1.1 * distance 

               
                
                self.cmd_vel.linear.x = linear_velocity 
                self.prev_distance = distance
               
            else:
                self.cmd_vel.linear.x = 0.0 
                
        self.cmd_vel.linear.x = min(self.cmd_vel.linear.x, self.MAX_LINEAR_VELOCITY)
        self.cmd_vel_pub.publish(self.cmd_vel)

        if abs(orr_error) < 0.04 and distance < 0.03:
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.cmd_vel)  
            print("Reached target position and heading",self.i)
            marker = self.create_marker(des_position[0], des_position[1], self.i,0.0,1.0,0.0)
            

            
            if marker is not None:
                self.marker_pub1.publish(marker)  
            self.i+=1
            return False  
        
        return True  
    def create_marker(self, x, y, id,r,g,b):
        """! Creates a marker for RViz visualization."""
        marker = Marker()
        
        marker.header.frame_id = "map"  
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1  
        marker.scale.x = 0.05  
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0  
        marker.color.r = r 
        marker.color.g = g  
        marker.color.b = b  

        # self.get_logger().info(f"Created marker: id={id}, position=({x}, {y})")
        
        return marker

    def detect_frontiers(self):
        """Detect frontiers in the map (unknown regions adjacent to free space)."""
        frontiers = []
        for y in range(1, self.height - 1):  # avoid borders
            for x in range(1, self.width - 1):
                if self.grid[y, x] == -1:  # Unknown space
                    # Check if it's adjacent to free space
                    if any(self.grid[y + dy, x + dx] == 0 for dx, dy in DIRECTIONS):
                        frontiers.append((x, y))
        return frontiers

    def plan_path_to_frontier(self, target):
        """Use A* to plan a path to the frontier."""
        # start = (self.width // 2, self.height // 2)  # Assume robot starts at the center
        y,x=self.convert_to_grid_coordinates(self.transformed_position.x,self.transformed_position.y)
        print(x,y,self.transformed_position)
        start=(int(x),int(y))
        astar = AStar(self.grid, self.resolution, self.origin_x, self.origin_y)
        return astar.find_path(start, target)

    def visualize_path(self, path):
        """Visualize the found path on the grid."""
        if path:
            fig, ax = plt.subplots()  # Make the plot bigger if needed
            
            # Create a custom color map:
            #  - Black for obstacles (100)
            #  - White for free space (0)
            #  - Gray for unknown (values -1)
            cmap = plt.cm.get_cmap('gray')  # Start with the 'gray' colormap
            cmap.set_under('gray')  # For unknown (-1) areas, use gray
            cmap.set_over('black')  # For obstacles (100), use black
            cmap.set_bad('white')  # For free space (0), use white
            
            # Display the grid using imshow
            ax.imshow(self.grid, cmap='gray', vmin=-1, vmax=100)  # Ensure the full range is covered
            
            ax.set_title('Occupancy Grid Map with Path')

            # Plot the path in red
            path_x, path_y = zip(*path)  # Unzip the path into x and y coordinates
            ax.plot(path_y, path_x, color='red', linewidth=2, marker='o', markersize=5)

            # Flip the y-axis to match ROS 2 RViz (origin at bottom-left)
            ax.invert_yaxis()

            # Hide the axis
            ax.axis('off')
            plt.show()
            # Debug: Show the plot
            print("Displaying path on map...")
            # plt.close(fig)
            # plt.show(block=False)  # Non-blocking show, allowing for continuous updates

            # sleep(1)  # Allow the plot to be visible for a second before updating the next one

            # plt.close(fig)  # Close the previous plot to make way for the new one
    
    def expand_obstacles(self, expansion_radius=4):
        """Expand obstacles by setting neighboring cells to 100."""
        new_grid = self.grid.copy()
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 100:  # If it's an obstacle
                    # Expand the obstacle by setting nearby cells to 100
                    for dy in range(-expansion_radius, expansion_radius + 1):
                        for dx in range(-expansion_radius, expansion_radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                new_grid[ny, nx] = 100  # Mark the surrounding area as obstacle
        self.grid = new_grid

    def convert_to_real_coordinates(self, grid_x, grid_y):
        """Convert grid coordinates to real-world coordinates."""
        real_x = self.origin_x + grid_y * self.resolution
        real_y = self.origin_y + grid_x * self.resolution
        return real_x, real_y
    def convert_to_real_coordinates_path(self, path):
        """Convert grid path to real-world coordinates (meters)."""
        real_world_path = []
        for grid_pos in path:
            grid_x, grid_y = grid_pos
            real_x = self.origin_x + grid_y * self.resolution  # y corresponds to column index (horizontal)
            real_y = self.origin_y + grid_x * self.resolution  # x corresponds to row index (vertical)
            real_world_path.append((real_x, real_y))
        return real_world_path
    def convert_to_grid_coordinates(self, real_x, real_y):
        """Convert real-world coordinates to grid coordinates."""
        grid_y=(real_x -self.origin_x)/self.resolution   
        grid_x=(real_y - self.origin_y ) / self.resolution
        return grid_y, grid_x


class AStar:
    def __init__(self, grid, resolution, origin_x, origin_y):
        self.grid = grid
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

    def heuristic(self, start, goal):
        """Euclidean distance heuristic"""
        x1, y1 = start
        x2, y2 = goal
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def manhattan_heuristic(self, start, goal):
        """Manhattan distance heuristic"""
        x1, y1 = start
        x2, y2 = goal
        return abs(x2 - x1) + abs(y2 - y1)

    def is_valid(self, node):
        """Check if the node is within the grid and not an obstacle"""
        x, y = node
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
            return self.grid[x, y] != 100  # 100 is an obstacle
        return False

    def reconstruct_path(self, came_from, current):
        """Reconstruct the path from start to goal"""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        return path[::-1]  # Return reversed path

    def find_path(self, start, goal):
        """A* algorithm to find the shortest path"""
        open_list = []
        heapq.heappush(open_list, (0 + self.heuristic(start, goal), 0, start))  # (f, g, position)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_list:
            _, current_g, current = heapq.heappop(open_list)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for direction in DIRECTIONS:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if self.is_valid(neighbor):
                    tentative_g_score = current_g + 1  # Each step has a cost of 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                        heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))

        return []  # No path found

def run_in_thread(frontier_explorer):
    """Run the frontier exploration in a separate thread."""
    frontier_explorer.get_logger().info("Starting the run method in a separate thread.")
    frontier_explorer.run()


import threading
def main(args=None):
    rclpy.init(args=args)
    frontier_explorer = FrontierExplorer()

    try:
        # Start the run method in a separate thread
        run_thread = threading.Thread(target=run_in_thread, args=(frontier_explorer,))
        run_thread.start()

        # Spin the node to handle odometry and map updates
        rclpy.spin(frontier_explorer)
        # print("dsssssssssssssssssssssssssss")
        
    except KeyboardInterrupt:
        pass
    finally:
        frontier_explorer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
