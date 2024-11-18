#!/usr/bin/env python3

import sys
import os
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
import numpy as np
from copy import copy
from PIL import Image, ImageOps 
import tf_transformations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from visualization_msgs.msg import Marker
import yaml
import pandas as pd

from copy import copy, deepcopy
import time

from graphviz import Graph

class Queue():
    def __init__(self, init_queue = []):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue)-1
    
    def __len__(self):
        numel = len(self.queue)
        return numel
    
    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if(i == self.start):
                tmpstr += "<"
                flag = True
            if(i == self.end):
                tmpstr += ">"
                flag = True
            
            if(flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'
            
        return tmpstr
    
    def __call__(self):
        return self.queue
    
    def initialize_queue(self,init_queue = []):
        self.queue = copy(init_queue)
    
    def sort(self,key=str.lower):
        self.queue = sorted(self.queue,key=key)
        
    def push(self,data):
        self.queue.append(data)
        self.end += 1
    
    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue)-1
        return p
    
class Nodee():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []
        
    def __repr__(self):
        return self.name
        
    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)
    
class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph('G')
    
    def __call__(self):
        for name,node in self.g.items():
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual
    
    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name
            
    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False
    
    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True   
        


class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)
    
    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""
        
    def __open_map(self,map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        map_name = map_df.image[0]
        im = Image.open(map_name)
        # size = 210, 297
        # im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]
        print(xmin,xmax,ymin,ymax,im.size)
        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array

class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        # print('ss',self.map.image_array.shape)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)
    
    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and 
            (i < map_array.shape[0]) and 
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value 
    
    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)
        
    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r
                
    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Nodee('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:                    
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left 
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left 
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])                    
        
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm
    
    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m
    
    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array

class AStar:

    def __init__(self, in_tree):
        self.q = Queue()
        self.dist = {name: np.inf for name, node in in_tree.g.items()}
        self.h = {name: 0 for name, node in in_tree.g.items()}
        self.via = {name: None for name, node in in_tree.g.items()}
        self.visited = {name: False for name, node in in_tree.g.items()}
        
        for name, node in in_tree.g.items():
           
            start = tuple(map(int, name.split(',')))
            end = tuple(map(int, in_tree.end.split(',')))
            self.h[name] = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            self.q.push(node)

    def __get_f_score(self, node):
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        self.dist[sn.name] = 0
        while len(self.q) > 0:
            
            self.q.sort(key=self.__get_f_score)
            u = self.q.pop()
            if u.name == en.name:
                break
            for i in range(len(u.children)):
                c = u.children[i]
                w = u.weight[i]
                new_dist = self.dist[u.name] + w
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name

    def reconstruct_path(self, sn, en):
        path = []
        dist = self.dist[en.name]
        current = en.name
        while current != sn.name:
            path.append(current)
            current = self.via[current]
        path.append(sn.name)
        path.reverse()
        return path, dist


def read_map_parameters(yaml_file):
    """Read map parameters from a YAML file."""
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data['resolution'], data['origin']

def convert_to_meters(path, resolution, origin):
    """Convert grid path to real-world coordinates."""
    real_path = []
    for coord in path:
        y, x = map(int, coord.split(','))
        real_x = origin[0] + (x * resolution)
        real_y = origin[1] - (y * resolution)
        real_path.append((real_x, real_y))
    return real_path


class Navigation(Node):
    """! Navigation node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        super().__init__(node_name)
        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.goal_received = False
        self.i=0
        self.MAX_LINEAR_VELOCITY = 1.5 
        self.kp_ang = 0.9
        self.ki_ang = 0.001
        self.kd_ang = 0.001
        self.prev_orr_error = 0.0
        self.integral_orr_error = 0.0
        
        
        self.kp_lin = 1.0
        self.ki_lin = 0.001
        self.kd_lin = 0.001
        self.prev_distance = 0.0
        self.integral_distance = 0.0

        self.time_step=0.1

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 1)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.marker_pub = self.create_publisher(Marker, 'robot_path', 10)
        self.marker_pub1 = self.create_publisher(Marker, 'planned_path', 10)
        self.cmd_vel = Twist()
        # Node rate
        self.rate = self.create_rate(1)

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.goal_received = True
        # self.get_logger().info(
        #     'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
    def wait_for_goal_pose(self):
        """! Block until the goal pose is received."""
        while not self.goal_received:
            
            rclpy.spin_once(self, timeout_sec=0.001)

        return self.goal_pose.pose.position.x,self.goal_pose.pose.position.y
    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose = data.pose
        
        # self.get_logger().info(
        #     'ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))

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


    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        
        vehicle_position = vehicle_pose.pose.position
        vehicle_position_values = (vehicle_position.x, vehicle_position.y, vehicle_position.z)
        vehicle_orientation = vehicle_pose.pose.orientation
        vehicle_orientation_values = (vehicle_orientation.x, vehicle_orientation.y, vehicle_orientation.z, vehicle_orientation.w)
        speed = current_goal_pose
       
        heading = np.arctan2((current_goal_pose[1]-vehicle_position_values[1]),(current_goal_pose[0]-vehicle_position_values[0])) 
     
        
        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Des speed.
        @param  heading   Des yaw angle.
        @return path      object containing the sequence of waypoints of the created path.

        """
        
        des_heading=heading
        des_position=speed
        vehicle_orientation=self.ttbot_pose.pose.orientation
        vehicle_orientation_values = (vehicle_orientation.x, vehicle_orientation.y, vehicle_orientation.z, vehicle_orientation.w)
        current_heading=tf_transformations.euler_from_quaternion(vehicle_orientation_values)

        
        orr_error = (des_heading - current_heading[2] + np.pi) % (2 * np.pi) - np.pi
        pos_error_x = des_position[0] - self.ttbot_pose.pose.position.x
        pos_error_y = des_position[1] - self.ttbot_pose.pose.position.y
        distance = np.sqrt(pos_error_x**2 + pos_error_y**2)

    

        if abs(orr_error) > 0.08: 
            orr_error_derivative = (orr_error - self.prev_orr_error) / self.time_step
            self.integral_orr_error += orr_error * self.time_step
            angular_velocity = (self.kp_ang * orr_error +
                                self.ki_ang * self.integral_orr_error +
                                self.kd_ang * orr_error_derivative)
            self.cmd_vel.linear.x = 0.0  
            self.cmd_vel.angular.z = angular_velocity
            self.prev_orr_error = orr_error
        else:
            self.cmd_vel.angular.z = 0.0  
             
            if distance > 0.05:
                
                distance_derivative = (distance - self.prev_distance) / self.time_step
                self.integral_distance += distance * self.time_step
                linear_velocity = (self.kp_lin * distance +
                                self.ki_lin * self.integral_distance +
                                self.kd_lin * distance_derivative)

               
                
                self.cmd_vel.linear.x = linear_velocity 
                self.prev_distance = distance
               
            else:
                self.cmd_vel.linear.x = 0.0 
                
        self.cmd_vel.linear.x = min(self.cmd_vel.linear.x, self.MAX_LINEAR_VELOCITY)
        self.cmd_vel_pub.publish(self.cmd_vel)

        if abs(orr_error) < 0.08 and distance < 0.05:
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

    def run(self,path):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """
        while rclpy.ok():
            
            self.a=True
           
            current_goal = path[self.i]
           
            
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            
            while self.a==True:
                rclpy.spin_once(self, timeout_sec=0.001) 

                
                self.a=self.move_ttbot(speed, heading)
            if len(path)==self.i:
                print("Completed")
                break

    def path_ploting(self,path):
        
        for j in range(len(path)):
            current_goal = path[j]
            
            marker = self.create_marker(current_goal[0], current_goal[1], j,1.0,0.0,0.0)
            # if marker is not None:
            self.marker_pub.publish(marker)
            time.sleep(0.1) 






def main(args=None):
    rclpy.init(args=args)
    mp = MapProcessor('/home/ros2/turttle/src/sim_ws/src/task_7/maps/maps/sync_classroom_map')
    map = Map('/home/ros2/turttle/src/sim_ws/src/task_7/maps/maps/sync_classroom_map')
    print(map.limits)
    kr = mp.rect_kernel(9,1)
  
    mp.inflate_map(kr,True)

    mp.get_graph_from_map()

    

    yaml_file = '/home/ros2/turttle/src/sim_ws/src/task_7/maps/maps/sync_classroom_map.yaml'  

    resolution, origin = read_map_parameters(yaml_file)
   
    


    nav = Navigation(node_name='Navigation')

    try:
        print('please click the goal pose in rviz')
        goalx,goaly=nav.wait_for_goal_pose()
        print('goal recieved. Planning ...............')
        start_x = (0.0 + origin[1]) / resolution
        start_y = (0.0 - origin[0]) / resolution

        mp.map_graph.root = f"{int(start_x)},{int(start_y)}"
        
        x = (-goaly + origin[1]) / resolution
        y = (goalx - origin[0]) / resolution

        mp.map_graph.end = f"{int(x)},{int(y)}"
      
        as_maze = AStar(mp.map_graph)

        start = time.time()
        as_maze.solve(mp.map_graph.g[mp.map_graph.root],mp.map_graph.g[mp.map_graph.end])
        end = time.time()
        print('Elapsed Time: %.3f'%(end - start))

        path_as,dist_as = as_maze.reconstruct_path(mp.map_graph.g[mp.map_graph.root],mp.map_graph.g[mp.map_graph.end])
        
        path_arr_as = mp.draw_path(path_as)

        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
        print(path_arr_as)
        
        ax.imshow(path_arr_as)
        ax.set_title('Path A*')

        plt.show()

        real_path = convert_to_meters(path_as, resolution, origin)
        nav.path_ploting(real_path)
        nav.run(real_path)
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()