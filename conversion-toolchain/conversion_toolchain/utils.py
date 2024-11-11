import json
from enum import Enum
from glob import glob
from os.path import *

import numpy as np
import onnx
from PIL import Image
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.model_translator.exceptions import ParsingWithRecommendationException

__here__ = dirname(__file__)

chosen_hw_arch = 'hailo8'
onnx_model_name = 'model'


class STR(str, Enum):
    MEANS = 'means'
    STDS = 'stds'
    HEIGHT = 'height'
    WIDTH = 'width'
    CHANNELS = 'channels'
    NCHW = 'nchw'


def calib_dataset(io_info: dict, calibration_images: list):
    means = io_info[STR.MEANS]
    stds = io_info[STR.STDS]
    width = io_info[STR.WIDTH]
    height = io_info[STR.HEIGHT]
    channels = io_info[STR.CHANNELS]
    nchw = io_info[STR.NCHW]

    grayscale = channels == 1

    if len(calibration_images) == 0:
        print('No calibration images provided, using default images.')
        images = glob(join(__here__, '**', '*.jpg'), recursive=True)
    else:
        images = calibration_images
    images = [Image.open(img) for img in images]
    images = [img.convert('L' if grayscale else 'RGB') for img in images]
    images = [img.resize((width, height)) for img in images]
    images = [np.array(img).astype('float32') for img in images]
    images = np.array(images)
    if grayscale:
        images = np.expand_dims(images, axis=-1)
    if nchw:
        images = images.transpose((0, 3, 1, 2))
    images = (images - means) / stds
    return images


def convert_to_hailo_onnx(input_path: str, output_path: str, input_json, calibration_images, logs):
    logs.add_message('Starting conversion', {'Target Hardware': chosen_hw_arch})
    runner = ClientRunner(hw_arch=chosen_hw_arch)
    try:
        start_node_names = input_json.pop('start_node_names', [])
        end_node_names = input_json.pop('end_node_names', [])
        start_node_names = start_node_names if start_node_names else None
        end_node_names = end_node_names if end_node_names else None

        runner.translate_onnx_model(input_path, onnx_model_name,
                                    start_node_names=start_node_names,
                                    end_node_names=end_node_names,
                                    )
    except ParsingWithRecommendationException as e:
        raise Exception(e)
    logs.add_data(**{'Translation': 'Done'})

    runner.optimize(calib_dataset(input_json, calibration_images))
    logs.add_data(**{'Optimizing Model': 'Done'})

    runner.compile()  # the returned HEF is not needed when working with ONNXRT
    logs.add_data(**{'Compiling Model': 'Done'})

    onnx_model = runner.get_hailo_runtime_model()  # only possible on a compiled model
    onnx.save(onnx_model, output_path)  # save model to file


def extract_io_info(onnx_path: str):
    graph = onnx.load(onnx_path).graph

    # means and stds from doc_string
    try:
        string = graph.doc_string
        js = json.loads(string)
        means = js['means']
        stds = js['vars']
    except Exception as e:
        means = [0]
        stds = [1]

    # get image input (shape==4)
    inputs = graph.input
    image_input = [i for i in inputs if len(i.type.tensor_type.shape.dim) == 4]
    if len(image_input) == 0:
        raise ValueError('No image input found for the ONNX')
    elif len(image_input) > 1:
        raise ValueError('More than one image input found')
    else:
        image_input = image_input[0]

    input_shape = [d.dim_value for d in image_input.type.tensor_type.shape.dim]
    # check if image is nchw or nhwc
    if input_shape[1] <= 3:
        nchw = True
    elif input_shape[1] > 3:
        nchw = False
    if nchw:
        channels, height, width = input_shape[1:]
    else:
        height, width, channels = input_shape[1:]

    # return json with io_info
    return {
        STR.MEANS: means,
        STR.STDS: stds,
        STR.HEIGHT: height,
        STR.WIDTH: width,
        STR.CHANNELS: channels,
        STR.NCHW: nchw
    }


def unzip_file(zip_path, output_dir):
    """Unzips a file and returns the paths to the ONNX file, JSON dict, and calibration images.

    Args:
        zip_path (str): Path to the zip file.
        output_dir (str): Directory to save the unzipped files.

    Returns:
        Tuple[str, str, List[str]]: Paths to the ONNX file, JSON dict, and calibration images.

    Raises:
        FileNotFoundError: If the ONNX file, JSON file, or calibration images are not found in the zip.
        ValueError: If more than one ONNX file, or JSON file are found in the zip.
    """
    from os import makedirs
    from os.path import join, exists
    from shutil import rmtree
    from zipfile import ZipFile
    from glob import glob

    if exists(output_dir):
        rmtree(output_dir)
    makedirs(output_dir, exist_ok=False)

    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Find the ONNX file
    onnx_path = glob(join(output_dir, '*.onnx'))
    if len(onnx_path) == 0:
        raise FileNotFoundError('No ONNX file found in the zip')
    elif len(onnx_path) > 1:
        raise ValueError('More than one ONNX file found in the zip')
    onnx_path = onnx_path[0]

    # Find the JSON file
    json_path = glob(join(output_dir, '*.json'))
    if len(json_path) == 0:
        raise FileNotFoundError('No JSON file found in the zip')
    elif len(json_path) > 1:
        raise ValueError('More than one JSON file found in the zip')
    json_path = json_path[0]
    with open(json_path, 'r') as f:
        input_js = json.load(f)

    # Find calibration images
    calib_images = glob(join(output_dir, '**', '*.jpg'), recursive=True)
    return onnx_path, input_js, calib_images
