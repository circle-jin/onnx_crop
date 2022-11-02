# onnx_crop

## Introduction
onnx 모델과 json 파일을 입력으로 받아서 모델을 crop, pow->Mul 변환해 주는 repo입니다.
현재 지원 되는 기능은 end-crop, pow-mul 입니다.

end-crop : 입력으로 받은 기준 노드의 뒤에 있는 노드들을 crop하고 새로운 output을 생성합니다.
pow-mul : pow로 2제곱 하고있는 노드를 mul로 변환합니다. (보드에서 pow를 지원하지 않을 시 사용)

## How to use
end-crop 사용 방법 : python3 onnx_modify.py -c ./json/endcrop_config_file.json
pow-mul 사용 방법 : python3 onnx_modify.py -c ./json/powmul_config_file.json


필요한 라이브러리
```bash
pip install -r requirements.txt
```