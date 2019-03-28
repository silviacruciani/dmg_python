import numpy as np
import math
import sys
from pycaster import pycaster
import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d
import scipy.spatial

class DexterousManipulationGraph():
    """class to read a DMG from files and do basic search"""
    def __init__(self):
        self._adjacency_list = None
        self._node_to_component = None
        self._node_to_position = None
        self._component_to_normal = None
        self._component_to_nodes = None
        self._node_to_angles = None
        self._supervoxel_angle_to_angular_component = None

        # data structures for the nearest neighbor
        self._kdtree = None
        self._idx_to_supervoxel = None
        self._supervoxel_to_angular_component = None

        #stuff for opposite finger component
        self._caster = None
        self._object_shape_file = None

        #length of the gripper's finger 
        self._finger_length = 0.04 #Yumi's finger

    def set_object_shape_file(self, filename):
        '''read the object shape file'''
        self._object_shape_file = filename
        self._caster = pycaster.rayCaster.fromSTL(filename, scale=1.0) #the object used to be in mm

    def read_nodes(self, filename):
        '''reads the Cartesian positions of all the nodes'''
        nodes_to_position = dict()
        positions_list = list()
        f = open(filename, 'r')
        idx = 0
        idx_to_sv = dict()
        for x in f:
            y = x.split()
            p = np.array([float(y[2]), float(y[3]), float(y[4])])
            n = (int(y[0]), int(y[1]))
            nodes_to_position[n] = p
            positions_list.append(p)
            idx_to_sv[idx] = n[0]
            idx += 1
            
        self._node_to_position = nodes_to_position
        self._idx_to_supervoxel = idx_to_sv
        self._kdtree = scipy.spatial.KDTree(positions_list)

    def read_graph(self, filename):
        '''reads the adjacency list'''
        nodes_to_list_of_nodes = dict()
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            node = (int(y[0]), int(y[1]))
            nodes_to_list_of_nodes[node]=[]
            for i in range(2, len(y), 2):
                nodes_to_list_of_nodes[node] += [(int(y[i]), int(y[i+1]))]
        self._adjacency_list = nodes_to_list_of_nodes

    def read_node_to_component(self, filename):
        '''reads the mapping from node id to connected component'''
        nodes_to_component = dict()
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            node = (int(y[0]), int(y[1]))
            nodes_to_component[node] = int(y[2])
        self._node_to_component = nodes_to_component

    def read_component_to_normal(self, filename):
        '''reads the normal associated to each component'''
        component_to_normal = dict()
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            component = int(y[0])
            normal = np.array([float(y[1]), float(y[2]), float(y[3])])
            component_to_normal[component] = normal;
        self._component_to_normal = component_to_normal

    def read_node_to_angles(self, filename):
        '''reads the admissible angles in one node. These nodes are in degrees!'''
        nodes_to_angles = dict()
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            node = (int(y[0]), int(y[1]))
            angles_list = list()
            for i in range(2, len(y)):
                angles_list += [int(y[i])]
            nodes_to_angles[node] = angles_list
        self._node_to_angles = nodes_to_angles

    def read_supervoxel_angle_to_angular_component(self, filename):
        '''reads the mapping from supervoxel id and angle to the node angular component'''
        node_angle_to_angle_component = dict()
        sv_to_angle_component = dict()
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            supervoxel_id = int(y[0])
            if not node_angle_to_angle_component.has_key(supervoxel_id):
                node_angle_to_angle_component[supervoxel_id] = dict()
                sv_to_angle_component[supervoxel_id] = set()
            angle = int(y[1])
            node_angle_to_angle_component[supervoxel_id][angle] = int(y[2])
            sv_to_angle_component[supervoxel_id].add(int(y[2]))
        self._supervoxel_angle_to_angular_component = node_angle_to_angle_component
        self._supervoxel_to_angular_component = sv_to_angle_component

    def get_zero_angle_axis(self, nx):
        '''given the component normal nx, returns the ny axis that is the axis corresponding to an angle of 0 in the ny nz plane'''
        a = 0
        b = 0
        c = 0
        if abs(nx[0]>0.000001):
            b = 0
            c = math.sqrt(nx[0]**2/(nx[0]**2 + nx[2]**2))
            a = -nx[2]**2/nx[0]
        elif abs(nx[1]>0.000001):
            c = 0;
            a = math.sqrt(nx[1]**2/(nx[1]**2 + nx[0]**2))
            b = -nx[0]**2/nx[1]
        else:
            a = 0;
            b = math.sqrt(nx[2]**2/(nx[2]**2 + nx[1]**2))
            c = -nx[1]**2/nx[2]
        ny = np.array([a, b, c])
        return ny

    def get_closest_nodes(self, point):
        '''returns the nodes whose center is the closest to the given point. They can be two if there are nodes with different angular component'''
        (_, idx) = self._kdtree.query(point)
        supervoxel = self._idx_to_supervoxel[idx]
        #this supervoxel can correspond to more than one node
        angular_components = self._supervoxel_to_angular_component[supervoxel]
        nodes = list()
        for c in angular_components:
            nodes.append((supervoxel, c))
        return nodes


    def ray_shape_intersections(self, start_point, goal_point):
        '''get the intersection of a ray with the object's shape (used for checks on the opposite finger)'''
        #important thing: the mesh is in mm, while the graph points are in m!
        source = 1000.0*start_point
        destination = 1000.0*goal_point
        intersections = self._caster.castRay(source, destination)
        intersection_points = list()
        for x in intersections:
            intersection_points.append(np.array(x)/1000.0)
        return intersection_points

    def get_opposite_nodes(self, node):
        '''get the nodes that are on the opposite face of the object'''
        start_point = self._node_to_position[node]
        normal = self._component_to_normal[self._node_to_component[node]]
        goal_point = start_point - 10*normal #10 m
        points = self.ray_shape_intersections(start_point, goal_point)
        if len(points) < 1:
            #check in the other case, i.e. the normal is pointing inside the object
            goal_point = start_point + 10*normal #10 m
            points = self.ray_shape_intersections(start_point, goal_point)
            if len(points) < 1:
                return None
            if len(points) == 1 and np.linalg.norm(points[0]-start_point) < 0.001:
                return None
        elif len(points) == 1 and np.linalg.norm(points[0]-start_point) < 0.001:
            goal_point = start_point + 10*normal #10 m
            points = self.ray_shape_intersections(start_point, goal_point)
            if len(points) < 1:
                return None
            if len(points) == 1 and np.linalg.norm(points[0]-start_point) < 0.001:
                return None

        #now we check if the first point corresponds to the starting point, and if yes we remove it
        if np.linalg.norm(points[0]-start_point) < 0.001:
            points = points[1:]

        #now get the nodes corresponding to the points
        opposite_nodes = list()
        for p in points:
            node_list = self.get_closest_nodes(p)
            opposite_nodes += node_list
        return opposite_nodes 


    def get_shortest_path(self, start, goal):
        '''finds the shortest path in only one component'''
        #sanity check on the component (if they are different no in-hand path can be found)
        if self._node_to_component[start] != self._node_to_component[goal]:
            return None
        #check what should happen in the fingers in the opposite component
        opposite_start_list = self.get_opposite_nodes(start)
        opposite_goal_list = self.get_opposite_nodes(goal)
        #get the desired component
        opposite_component_start = set()
        opposite_component_goal = set()
        for n in opposite_start_list:
            opposite_component_start.add(self._node_to_component[n])
        for n in opposite_goal_list:
            opposite_component_goal.add(self._node_to_component[n])
        admissible_opposite_components = opposite_component_start & opposite_component_goal
        if len(admissible_opposite_components) < 1:
            return None

        #initialization for Dijkstra
        distances = dict()
        q = self._adjacency_list.keys()
        prev = dict()
        for x in self._adjacency_list:
            distances[x] = sys.float_info[0]
        distances[start] = 0.0
        prev[start] = None
        
        #for the distance between the nodes, for now use simple Euclidian distance
        def nodes_distance(node1, node2):
            point1 = np.array(self._node_to_position[node1])
            point2 = np.array(self._node_to_position[node2])
            return np.linalg.norm(point1-point2)

        while len(q) > 0:
            current_node = min(q, key=lambda node: distances[node])
            if current_node == goal:
                break
            if distances[current_node] == sys.float_info[0]:
                return None #no path can be found
            q.remove(current_node)
            #get the neighbors
            neighbor_nodes = self._adjacency_list[current_node]
            for n in neighbor_nodes:
                new_d = distances[current_node] + nodes_distance(n, current_node)
                #check if the opposite finger is valid
                valid = False
                opposite_nodes = self.get_opposite_nodes(n)
                for opposite_n in opposite_nodes:
                    if self._node_to_component[opposite_n] in admissible_opposite_components:
                        valid = True
                if not valid:
                    new_d = sys.float_info[0]

                if new_d < distances[n]:
                    distances[n] = new_d
                    prev[n] = current_node
                    
        path = [goal]
        if prev[goal] is None:
            return path
        prev_node = prev[goal]
        path.append(prev_node)
        while prev[prev_node] is not None:
            prev_node = prev[prev_node]
            path.append(prev_node)
        return path

    def get_rotations(self, start_angle, goal_angle, path):
        '''return a sequence of angles to associate with the given path'''
        cc = self._node_to_component[path[0]]
        normal = self._component_to_normal[cc]
        zero_axis = self.get_zero_angle_axis(normal)
        planar_zero_axis = -zero_axis[0:2]
        planar_zero_axis = planar_zero_axis/np.linalg.norm(planar_zero_axis)
        
        angles = [start_angle]
        angle = start_angle
        for i in range(len(path)-1, 0, -1):
            prev_node = path[i]
            next_node = path[i-1]
            next_valid_angles = self._node_to_angles[next_node]
            if angle in next_valid_angles:
                angles += [angle]
            else:
                #get an angle in common between the two (if there's the goal angle, put that)
                prev_valid_angles = self._node_to_angles[prev_node]
                angles_in_common = list(set(prev_valid_angles) & set(next_valid_angles))
                if goal_angle in angles_in_common:
                    next_angle = goal_angle
                else:
                    #get the angle closest to what we already have
                    _ , next_angle = min(enumerate(angles_in_common), key=lambda x: abs(x[1]-angle))
                angles += [next_angle]
                angle = next_angle
        angles += [goal_angle]
        return angles

    def plot_graph(self):
        '''Use to visualize the shape and the graph'''
        object_mesh = mesh.Mesh.from_file(self._object_shape_file)
        self._figure = plt.figure()
        self._axes = mplot3d.Axes3D(self._figure)
        self._axes.set_xlabel('x [mm]')
        self._axes.set_ylabel('y [mm]')
        self._axes.set_zlabel('z [mm]')
        self._axes.add_collection3d(mplot3d.art3d.Poly3DCollection(object_mesh.vectors))
        scale = object_mesh.points.flatten(-1)
        self._axes.auto_scale_xyz(scale, scale, scale)
        for n in self._adjacency_list:
            point1 = 1000*self._node_to_position[n]
            for m in self._adjacency_list[n]:
                point2 = 1000*self._node_to_position[m]
                xline = np.array([point1[0], point2[0]])
                yline = np.array([point1[1], point2[1]])
                zline = np.array([point1[2], point2[2]])
                self._axes.plot3D(xline, yline, zline, 'gray')
        self._figure.show()

    def plot_path(self, path, angles, color='yellow'):
        '''use to visualize the given path'''
        for i in range(len(path)-1, 0, -1):
            node1 = path[i]
            node2 = path[i-1]
            point1 = 1000*self._node_to_position[node1]
            point2 = 1000*self._node_to_position[node2]
            xline = np.array([point1[0], point2[0]])
            yline = np.array([point1[1], point2[1]])
            zline = np.array([point1[2], point2[2]])
            self._axes.plot3D(xline, yline, zline, color)
            #plot the current fingers
            curr_angle = angles[len(angles)-i-1]
            self.plot_finger(node2, curr_angle, color)
            self.plot_finger(node1, curr_angle, color)
        self._figure.show()

    # def axis_angle_matrix(axis, angle):
    #     '''returns '''
    #     def skew_symmetric(axis):



    def plot_finger(self, node, angle, color = 'yellow'):
        '''use to plot a finger in a given configuration'''
        point = self._node_to_position[node]    
        #the zero axis is opposite to the finger axis in the convention
        normal = self._component_to_normal[self._node_to_component[node]]
        zero_axis = self.get_zero_angle_axis(normal)      
        #rotate the zero angle axis by the angle
        theta = angle*math.pi/180
        #get the axis-angle matrix 
        rrt = np.outer(normal, normal)
        Sr = np.array([[0, -normal[2], normal[1]], [normal[2], 0, -normal[0]], [-normal[1], normal[0], 0]])
        R = rrt + (np.identity(3) - rrt)*math.cos(theta) + Sr*math.sin(theta)
        #R = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]]
        finger_axis = self._finger_length*np.dot(R, zero_axis)
        xline = [1000*point[0], 1000*point[0] + 1000*finger_axis[0]]
        yline = [1000*point[1], 1000*point[1] + 1000*finger_axis[1]]
        zline = [1000*point[2], 1000*point[2] + 1000*finger_axis[2]] #assumed no variation in z. To be improved
        self._axes.plot3D(xline, yline, zline, color)
        self._figure.show()

    def visualize(self):
        '''use to visualize the figures'''
        plt.show()