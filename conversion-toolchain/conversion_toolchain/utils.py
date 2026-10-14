import json
from enum import Enum
from glob import glob
from os.path import *

import numpy as np
import onnx
from PIL import Image

# hailo_sdk_client is only available inside the Docker container.
# Import it lazily inside convert_to_hailo_onnx to keep the rest of
# this module importable without the SDK installed.

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

    # Reshape means/stds so they broadcast correctly for both NCHW and NHWC.
    means = np.array(means, dtype=np.float32)
    stds  = np.array(stds,  dtype=np.float32)
    if nchw and means.ndim == 1 and means.shape[0] > 1:
        means = means.reshape(-1, 1, 1)
        stds  = stds.reshape(-1, 1, 1)

    images = (images - means) / stds
    return images


def find_hailo_end_nodes(onnx_path: str) -> list:
    """
    Find the detection-head Conv outputs for a YOLO-family model.

    YOLO detection heads output through cv2/cv3 Conv layers that ultimately
    feed into Reshape nodes for the DFL decode step.  Hailo can't handle
    those Reshapes, so we stop just before them.

    Two naming conventions are handled:
    - Proper names (/model.N/cv*.*.2/Conv): direct Conv → Reshape
    - Generic names (Conv_X): Conv → Concat → Reshape (one extra hop)

    The last Reshape in the graph is the DFL decode output (problematic);
    it is excluded so we only collect the earlier detection-head Reshapes.
    """
    import re as _re
    model = onnx.load(onnx_path)

    # Identify the detection head block (highest /model.N/ index)
    max_model_idx = 0
    for node in model.graph.node:
        m = _re.search(r'/model\.(\d+)/', node.name or '')
        if m:
            max_model_idx = max(max_model_idx, int(m.group(1)))

    head_prefix = f'/model.{max_model_idx}/' if max_model_idx > 0 else None

    # Map each output tensor to its producing node
    tensor_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            tensor_to_node[out] = node

    def _conv_within_2hops(tensor):
        """Return the Conv node within 2 hops, or None."""
        p1 = tensor_to_node.get(tensor)
        if p1 is None:
            return None
        if p1.op_type == 'Conv':
            return p1
        # Second hop: follow the first data input of the intermediate node
        if p1.input:
            p2 = tensor_to_node.get(p1.input[0])
            if p2 and p2.op_type == 'Conv':
                return p2
        return None

    # Find all Reshape nodes ordered by position in the graph
    reshape_nodes = [(i, n) for i, n in enumerate(model.graph.node) if n.op_type == 'Reshape']
    if not reshape_nodes:
        return []

    # Exclude the LAST Reshape (DFL decode output — produces the problematic tensor)
    reshape_nodes_to_use = reshape_nodes[:-1]

    end_nodes, seen = [], set()
    for _, reshape in reshape_nodes_to_use:
        # For multi-input Reshapes (via Concat), check all Concat inputs
        data_input = reshape.input[0]
        intermediate = tensor_to_node.get(data_input)

        candidates = []
        if intermediate and intermediate.op_type == 'Conv':
            candidates = [intermediate]
        elif intermediate and intermediate.input:
            # The intermediate might be a Concat collecting multiple Conv outputs
            for inp in intermediate.input:
                conv = _conv_within_2hops(inp)
                if conv:
                    candidates.append(conv)
            if not candidates:
                # Fall back: single path
                conv = _conv_within_2hops(data_input)
                if conv:
                    candidates = [conv]

        for conv in candidates:
            if not conv.name or conv.name in seen:
                continue
            # Filter to detection head only when proper names are available
            if head_prefix and head_prefix not in conv.name:
                continue
            end_nodes.append(conv.name)
            seen.add(conv.name)

    return end_nodes


def convert_to_hailo_onnx(input_path: str, output_path: str, input_json, calibration_images, logs):
    from hailo_sdk_client import ClientRunner
    from hailo_sdk_client.model_translator.exceptions import ParsingWithRecommendationException

    logs.add_message('Starting conversion', {'Target Hardware': chosen_hw_arch})
    runner = ClientRunner(hw_arch=chosen_hw_arch)

    start_node_names = input_json.pop('start_node_names', []) or None
    end_node_names   = input_json.pop('end_node_names',   []) or None

    # If no end nodes supplied, auto-detect from the ONNX graph.
    if not end_node_names:
        end_node_names = find_hailo_end_nodes(input_path) or None
        if end_node_names:
            logs.add_message('Auto-detected end nodes', {'nodes': end_node_names})

    try:
        runner.translate_onnx_model(input_path, onnx_model_name,
                                    start_node_names=start_node_names,
                                    end_node_names=end_node_names)
    except ParsingWithRecommendationException as e:
        raise Exception(e)

    logs.add_data(**{'Translation': 'Done'})

    runner.optimize(calib_dataset(input_json, calibration_images))
    logs.add_data(**{'Optimizing Model': 'Done'})

    runner.compile()  # the returned HEF is not needed when working with ONNXRT
    logs.add_data(**{'Compiling Model': 'Done'})

    onnx_model = runner.get_hailo_runtime_model()  # only possible on a compiled model

    # Hailo ORT build supports up to opset 16; downgrade if necessary.
    from onnx import version_converter
    for opset in onnx_model.opset_import:
        if opset.domain == '' and opset.version > 16:
            onnx_model = version_converter.convert_version(onnx_model, 16)
            break

    onnx.save(onnx_model, output_path)


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

    # Find calibration images (accept common JPEG extensions)
    calib_images = []
    for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG'):
        calib_images.extend(glob(join(output_dir, '**', ext), recursive=True))
    return onnx_path, input_js, calib_images
