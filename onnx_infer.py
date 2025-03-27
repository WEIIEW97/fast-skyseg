from pathlib import Path
import copy
import gc

import cv2 as cv
import numpy as np
import onnxruntime

def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv.cvtColor(resize_image, cv.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype('uint8')

    return onnx_result

if __name__ == "__main__":
    data_path = Path("/home/william/extdisk/data/FC1/GY2ASH24GV0024/L")
    save_path = Path("/home/william/extdisk/data/FC1/GY2ASH24GV0024/seg")
    save_path.mkdir(exist_ok=True)
    onnx_session = onnxruntime.InferenceSession("/home/william/Downloads/skyseg.onnx")
    image_path = data_path.glob("*.png")
    for p in image_path:
        image = cv.imread(str(p))
        image_name = p.stem
        ori_h, ori_w = image.shape[:2]
        if(image.shape[0] >= 640 and image.shape[1] >= 640):
            image = cv.pyrDown(image)
        result_map = run_inference(onnx_session,[320,320],image)
        result_resize = cv.resize(result_map, (ori_w, ori_h), interpolation=cv.INTER_LINEAR)
        cv.imwrite(str(save_path / (image_name + ".png")), result_resize)

    print("done!")