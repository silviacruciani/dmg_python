from dmg_class import DexterousManipulationGraph
import os

def main():
    dmg = DexterousManipulationGraph()
    #read the dmg from files
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = file_path + '/dmg_files/'
    object_name = 'glue'
    dmg.set_object_shape_file(file_path + object_name + '/' + object_name + '.stl')
    dmg.read_graph(file_path + object_name + '/graph_' + object_name + '_12_20.txt')
    dmg.read_nodes(file_path + object_name + '/node_position_' + object_name + '_12_20.txt')
    dmg.read_node_to_component(file_path + object_name + '/node_component_' + object_name + '_12_20.txt')
    dmg.read_component_to_normal(file_path + object_name + '/component_normal_' + object_name + '_12_20.txt')
    dmg.read_node_to_angles(file_path + object_name + '/node_angle_' + object_name + '_12_20.txt')
    dmg.read_supervoxel_angle_to_angular_component(file_path + object_name  + '/node_angle_angle_component_' + object_name + '_12_20.txt')
    
    #nodes are a tuple (supervoxel id, angular component id)
    #for glue example
    start_node = (82, 0)
    goal_node = (52, 0)
    start_angle = 100
    goal_angle = 0    
    #for crayola example
    # start_node = (20, 0)
    # goal_node = (80, 0)
    # start_angle = 180
    # goal_angle = 100

    path = dmg.get_shortest_path(start_node, goal_node)
    angles = dmg.get_rotations(start_angle, goal_angle, path)
    dmg.plot_graph()
    dmg.plot_path(path, angles)
    #plot the start and the goal to make things clear
    dmg.plot_finger(start_node, start_angle, color='red')
    dmg.plot_finger(goal_node, goal_angle, color='green')
    dmg.visualize()

if __name__ == '__main__':
    main()