from dmg_class import DexterousManipulationGraph
import os

def main():
    dmg = DexterousManipulationGraph()
    #read the dmg from files
    file_path = os.path.dirname(os.path.abspath(__file__))
    dmg.set_object_shape_file(file_path + '/dmg_files/crayola.stl')
    dmg.read_graph(file_path + '/dmg_files/graph_crayola_12_20.txt')
    dmg.read_nodes(file_path + '/dmg_files/node_position_crayola_12_20.txt')
    dmg.read_node_to_component(file_path + '/dmg_files/node_component_crayola_12_20.txt')
    dmg.read_component_to_normal(file_path + '/dmg_files/component_normal_crayola_12_20.txt')
    dmg.read_node_to_angles(file_path + '/dmg_files/node_angle_crayola_12_20.txt')
    dmg.read_supervoxel_angle_to_angular_component(file_path + '/dmg_files/node_angle_angle_component_crayola_12_20.txt')
    
    #nodes are a tuple (supervoxel id, angular component id)
    start_node = (20, 0)
    goal_node = (80, 0)
    path = dmg.get_shortest_path(start_node, goal_node)
    dmg.plot_graph()
    dmg.plot_path(path)
    dmg.visualize()

if __name__ == '__main__':
    main()