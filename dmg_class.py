import numpy as np
import math
import sys
from pycaster import pycaster
import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d

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

        #stuff for opposite finger component
        self._caster = None
        self._object_shape_file = None

    def set_object_shape_file(self, filename):
        '''read the object shape file'''
        self._object_shape_file = filename
        self._caster = pycaster.rayCaster.fromSTL(filename, scale=1.0) #the object used to be in mm

    def read_nodes(self, filename):
        '''reads the Cartesian positions of all the nodes'''
        nodes_to_position=dict()
        f = open(filename, 'r')
        for x in f:
            y=x.split()
            nodes_to_position[(int(y[0]), int(y[1]))]=np.array([float(y[2]), float(y[3]), float(y[4])])
        self._node_to_position = nodes_to_position

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
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            supervoxel_id = int(y[0])
            if not node_angle_to_angle_component.has_key(supervoxel_id):
                node_angle_to_angle_component[supervoxel_id] = dict()
            angle = int(y[1])
            node_angle_to_angle_component[supervoxel_id][angle] = int(y[2])
        self._supervoxel_angle_to_angular_component = node_angle_to_angle_component

    def get_zero_angle_axis(self, nx):
        '''given the component normal nx, returns the ny axis that is the axis corresponding to an angle of 0 in the ny nz plane'''
        a = 0
        b = 0
        c = 0
        if abs(nx[0]>0.0000000001):
            a = 0
            b = math.sqrt(nx[0]**2/(nx[0]**2 + nx[2]**2))
            c = -nx[2]**2/nx[0]
        elif abs(nx[1]>0.00000000001):
            c = 0;
            a = math.sqrt(nx[1]**2/(nx[1]**2 + nx[0]**2))
            b = -nx[0]**2/nx[1]
        else:
            a = 0;
            b = math.sqrt(nx[2]**2/(nx[2]**2 + nx[1]**2))
            c = -nx[1]**2/nx[2]
        ny = np.array([a, b, c])
        return ny

    def ray_shape_intersections(self, start_point, goal_point):
        '''get the intersection of a ray with the object's shape (used for checks on the opposite finger)'''
        #important thing: the mesh is in mm, while the graph points are in m!
        source = 1000.0*start_point
        destination = 1000.0*goal_point
        intersections = self._caster.castRay(source, destination)
        intersection_points = list()
        for x in intersections:
            intersection_points.append(x/1000.0)
        return intersection_points

    def get_shortest_path(self, start, goal):
        '''finds the shortest path in only one component (disregards opposite finger)'''
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
                #TO DO: check if the opposite finger is valid
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

    def plot_path(self, path):
        '''use to visualize the given path'''
        for i in range(len(path)-1, 0, -1):
            node1 = path[i]
            node2 = path[i-1]
            point1 = 1000*self._node_to_position[node1]
            point2 = 1000*self._node_to_position[node2]
            xline = np.array([point1[0], point2[0]])
            yline = np.array([point1[1], point2[1]])
            zline = np.array([point1[2], point2[2]])
            self._axes.plot3D(xline, yline, zline, 'red')
        self._figure.show()

    def visualize(self):
        '''use to visualize the figures'''
        plt.show()