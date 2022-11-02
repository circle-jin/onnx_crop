import argparse
import onnx
from onnx import helper, checker, TensorProto
import json

global save_path
global node_map
global output_map

# graph의 노드를 index가 아닌 name으로 접근하기 위한 dict 생성
def createGraphMemberMap(graph_member_list):
    member_map=dict()
    for n in graph_member_list:
        member_map[n.name]=n
    return member_map

def print_json_information(config_json):
    if config_json is not None:
        print("[Input file information]")

        for key, value in config_json.items():
            print(f'   {key} : {value}')

# [Input file information]
#    PREFIX               : 1028
#    ONNX Model           : ./yolov5s_crop_post_processing.onnx
#    Prepost file         : ./UserConfig/prepost_yolov5s_crop_post_processing.yaml
#    Address mapping file : ./UserConfig/addrmap_in_yolov5s_crop_post_processing.yaml
#    Option : -CheckPrePost : ON
#    Option : -SaveOptOnnx
def save_onnx_model(onnx_model, save_path):
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, save_path)
    print("[Successful], saved_path : " + save_path)

# pow -> Mul로 변환 (2제곱으로 사용된 pow의 Mul 변환)
def convert_pow_to_mul(graph, onnx_model):
    print('------------------------------------------')
    print('[pow-mul] start')
    global save_path
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
    print('------------------------------------------')
    save_onnx_model(onnx_model, save_path)

# 리스트로 받은 타겟 이후의 노드들 전부 삭제
def crop_end_of_node(graph, onnx_model, crop_target, output_shape):
    print('------------------------------------------')
    print('[end-crop] start')

    global node_map
    global output_map
    global save_path

    # 기존의 output 삭제
    for key, value in output_map.items():
        graph.output.remove(value) 

    # 새로운 output 생성
    # target 이후의 노드들을 crop 하기 위해 target의 output을 find_key로 설정
    find_key = []
    for i in range(0, len(crop_target)):
        graph.output.extend([helper.make_tensor_value_info("out"+str(i), TensorProto.FLOAT, output_shape[i])])
        find_key.extend(node_map[crop_target[i]].output)

    # 노드의 input에 자른 node의 output이 있는 경우 crop
    # find_key에 자신의 output을 추가한 뒤, crop
    for key, value in node_map.items():
        for input_index in range(0, len(value.input)):
            if value.input[input_index] in find_key:
                find_key.extend(value.output)
                graph.node.remove(value)
                break

    # 끝 노드를 output을 수정 (새로운 output과 연결되게)
    for i in range(0, len(crop_target)):
        # attributes 필요 없는 노드라면
        if len(node_map[crop_target[i]].attribute) == 0:
            end_node = onnx.helper.make_node(
                name=node_map[crop_target[i]].name,
                op_type=node_map[crop_target[i]].op_type,
                inputs=node_map[crop_target[i]].input,
                outputs=["out" + str(i)],
            )
        # attributes 필요 있는 노드라면
        else:
            end_node = onnx.helper.make_node(
                name=node_map[crop_target[i]].name,
                op_type=node_map[crop_target[i]].op_type,
                inputs=node_map[crop_target[i]].input,
                outputs=["out" + str(i)],
                attributes=node_map[crop_target[i]].attribute
            ) 

        # 수정할 노드의 index 가져오기
        for j in range(0, len(graph.node)):
            if graph.node[j].name == crop_target[i]:
                graph.node.remove(graph.node[j])
                graph.node.insert(j, end_node)

    print('[end-crop] finish')
    print('------------------------------------------')
    save_onnx_model(onnx_model, save_path)

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=True)
    args = parser.add_argument_group('Options')
    # args.add_argument('-p', '--model_path', type=str, required=True,
    #                   help='Example: ../yolov5s.onnx')
    # args.add_argument('-sp', '--save_path', type=str, required=True,
    #                   help='Example: ../yolov5s.onnx')
    # args.add_argument('-m', '--mode', type=str,
    #                   help='Example: pow-mul, end-crop')
    # args.add_argument('-ect', '--end_crop_target',  type=str,
    #                 help='require_mode: end-crop, Example: Sigmoid_204,Sigmoid_223,Sigmoid_242')
    # args.add_argument('-oh', '--output_shape',  type=str,
    #                   help='require_mode: end-crop, Example: [1,3,80,80,85] [1,3,80,80,85]')
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

            # input_file의 정보 출력
            print_json_information(config)

            model_path = config.get('model_path')
            onnx_model = onnx.load(model_path)
            graph = onnx_model.graph

            # 수정 후 모델 저장 경로
            global save_path
            save_path = config.get('save_path')

            # node의 정보를 dict 형태로 저장(index 말고도 node에 접근하기 위해)
            global node_map
            global output_map

            node_map = createGraphMemberMap(graph.node)
            output_map = createGraphMemberMap(graph.output)


            if config.get('mode', None) == None:
                """
                if the argument is None
                """
                print("mode == None, usage: node_control.py [-h]")
            elif config.get('mode', None) == 'pow-mul':
                """
                pow를 mul로 변환하고 싶은 경우
                """
                convert_pow_to_mul(graph, onnx_model)
            elif config.get('mode', None) == 'end-crop':
                """
                모델의 끝부분만 자르는 경우
                """

                crop_target = config.get('end_crop_target', None) # 잘라야 하는 기준 설정, 이 노드 이후부터 crop
                output_shape = config.get('output_shape', None) # new_end_node와, new_output 노드를 연결시키기 위해 필요한 output_shape

                if crop_target == None or output_shape == None:
                    print("usage: node_control.py [-h]")
                    return

                crop_end_of_node(graph, onnx_model, crop_target, output_shape)

if __name__ == '__main__':
    main()