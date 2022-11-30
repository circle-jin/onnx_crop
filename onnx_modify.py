import argparse
import onnx
from onnx import helper, checker, TensorProto
import json
import numpy as np

global save_path
global node_map
global output_map

def createGraphMemberMap(graph_member_list):
    """Create a dict to access the attributes(e.g. node,  output) of the graph with a name rather than an index

    Args:
        graph_member_list (google.protobuf): Attributes of the model graph (e.g. node, output)

    Returns:
        dict: dict consisting of attributes of graph
    """
    member_map=dict()
    for n in graph_member_list:
        member_map[n.name]=n
    return member_map

def print_json_information(config_json):
    """Print information of the input JSON file

    Args:
        config_json (json): Contents of input JSON file
    """
    if config_json is not None:
        print('------------------------------------------')
        print("[Input file information]")

        for key, value in config_json.items():
            print(f'   {key} : {value}')
        print('------------------------------------------')

def save_onnx_model(onnx_model, save_path):
    """Save the model as ONNX

    Args:
        onnx_model (onnx.ModelProto): loaded onnx model
        save_path (str): Path to save onnx model
    """
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, save_path)
    print("[Successful], saved_path : " + save_path)

def modify_attribute_of_Conv(graph, onnx_model, modify_require):
    """Modify the attribute of the Conv
    Args:
        graph (onnx.ModelProto): loaded onnx model graph
        onnx_model (onnx.ModelProto): loaded onnx model
        modify_require (dict): conditions of attributes to be changed
    """
    print('[modify-attribute] start')
    global node_map
    global output_map

    # If you change kernel_shape, the variables needed to modify the weight rawdata
    weight_name_list = list()
    modify_weight_dims = modify_require.get('change_kernel_shape', None)

    target_node_op = modify_require['target_node_type']
    # iterator for finding node's op
    for i in range(0, len(graph.node)):
        # Conv 노드가 아닐 경우 continue
        if graph.node[i].op_type != target_node_op:
            continue
        new_attribute_list = list()
        j=0
        check_change = False

        # In the case of Conv node, 
        # Compared to the conditions, Check if there is an attribute to change
        while(True):
            if j == len(graph.node[i].attribute):
                break
            attribute_name = graph.node[i].attribute[j].name
            if attribute_name in modify_require['target_attribute_name']:
                target_attribute = modify_require["target_" + attribute_name]
                change_attribute = modify_require["change_" + attribute_name]

                original_attribute = graph.node[i].attribute[j].ints
                if original_attribute == target_attribute:
                    new_attribute_list.append(helper.make_attribute(attribute_name, change_attribute))
                    graph.node[i].attribute.pop(j)
                    j-=1
                    check_change = True
            j+=1
        # If you change the Original_attribute, Put new_attribute into the attribute of the existing Conv
        if check_change == True:
            if modify_weight_dims is not None:
                weight_name_list.append(graph.node[i].input[1])
            graph.node[i].attribute.extend(new_attribute_list)
            
    # In the case of Conv node
    # If you change the kernel_shape, change rawdata value of weight to match new kernel_shape ()
    if modify_weight_dims is not None:
        for i in range(0, len(graph.initializer)):
            if graph.initializer[i].name in weight_name_list:
                graph.initializer[i].dims[2:4] = modify_weight_dims

                numpy_data = np.random.rand(graph.initializer[i].dims[0],
                                            graph.initializer[i].dims[1], 
                                            graph.initializer[i].dims[2],
                                            graph.initializer[i].dims[3])
                numpy_data_type = modify_require.get('weight_type', None)
                if numpy_data_type == "float32" or numpy_data_type == None:
                    numpy_data = numpy_data.astype(np.float32)

                rawdata = numpy_data.tobytes()
                graph.initializer[i].raw_data = rawdata
    print('[modify-attribute] finish')

def convert_pow_to_mul(graph, onnx_model):
    """POW-> MUL conversion (Two-squared pow converted to Mul)

    Args:
        graph (onnx.ModelProto): loaded onnx model graph
        onnx_model (onnx.ModelProto): loaded onnx model
    """
    print('[pow-mul] start')
    index = []
    for i in range(0, len(graph.node)):
        if graph.node[i].name.find('Pow') == 0:
            index = i
            input_node = [graph.node[i].input[0], graph.node[i].input[0]]
            output_node = graph.node[i].output


            Mul_node = onnx.helper.make_node(
                name="Mul_"+str(index),  # Name is optional.
                op_type="Mul",
                inputs=input_node,
                outputs=output_node
            )
            graph.node.remove(graph.node[index]) 
            graph.node.insert(index, Mul_node)

    print('[pow-mul] finish')

def crop_end_of_node(graph, onnx_model, end_crop_node, output_shape):
    """Delete all nodes after the node of the condition

    Args:
        graph (onnx.ModelProto): loaded onnx model graph
        onnx_model (onnx.ModelProto): loaded onnx model
        end_crop_node (list): A list of names in nodes that must cut the end
        output_shape (list): Output shape of nodes that need to cut the end
    """
    print('[end-crop] start')

    global node_map
    global output_map

    # Delete the original output
    for key, value in output_map.items():
        graph.output.remove(value) 

    # Creating a New Output
    # find_key setting to find and erase the remaining nodes (nodes after END_NODE)
    # Find_KEY is a list of names that need to be deleted.
    find_key = []
    for i in range(0, len(end_crop_node)):
        graph.output.extend([helper.make_tensor_value_info("out"+str(i), TensorProto.FLOAT, output_shape[i])])
        find_key.extend(node_map[end_crop_node[i]].output)


    # If the input of the current node is in Find_Key
    # Add the output name of the current node to Find_KEY and delete the current node
    for key, value in node_map.items():
        for input_index in range(0, len(value.input)):
            if value.input[input_index] in find_key:
                find_key.extend(value.output)
                graph.node.remove(value)
                break

    # Connect the end node with the new output
    for i in range(0, len(end_crop_node)):
        for j in range(0, len(graph.node)):
            if graph.node[j].name == end_crop_node[i]:
                graph.node[j].output[0] = "out" + str(i)

    print('[end-crop] finish')

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=True)
    args = parser.add_argument_group('Options')

    args.add_argument(
        '-c',
        '--config_file',
        dest='config_file',
        type=str,
        default=None,
        help='json file is required as input. The example jsonfile is in the ./json',
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.config_file is not None:
        if '.json' in args.config_file:
            config = json.load(open(args.config_file))
            print_json_information(config)
            model_path = config.get('model_path')

            onnx_model = onnx.load(model_path)
            graph = onnx_model.graph

            # path to save the modified model
            global save_path
            save_path = config.get('save_path')

            # Graph's node and output information is created in the form of dictionary (to access node with name instead of index)
            global node_map
            global output_map
            node_map = createGraphMemberMap(graph.node)
            output_map = createGraphMemberMap(graph.output)
            

            if config.get('mode', None) == None:
                """
                if the argument is None
                """
                print("mode == None, usage: node_control.py [-h]")
                return

            if 'pow-mul' in config.get('mode', None):
                convert_pow_to_mul(graph, onnx_model)
            if 'end-crop' in config.get('mode', None):
                end_crop_config = config.get('end_crop', None)
                end_crop_node = end_crop_config.get('end_crop_node', None)
                output_shape = end_crop_config.get('output_shape', None)
                if end_crop_node == None or output_shape == None:
                    print("Enter end_crop information in json")
                    return
                crop_end_of_node(graph, onnx_model, end_crop_node, output_shape)
            if 'modify-attribute' in config.get('mode', None):
                modify_require = config.get('modify_conv', None)
                modify_attribute_of_Conv(graph, onnx_model, modify_require)
            save_onnx_model(onnx_model, save_path)

if __name__ == '__main__':
    main()