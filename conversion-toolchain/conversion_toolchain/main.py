import argparse
from os import makedirs
from os.path import split, splitext, join

from .logger import Logs
from .utils import convert_to_hailo_onnx, unzip_file


def cli():
    parser = argparse.ArgumentParser(description='Transpile ONNX to an optimized Hailo-ONNX')

    parser.add_argument('--zip-path', required=True, help='Path to the Zip file')
    parser.add_argument('--output-dir', required=True, help='Directory to save the Hailo-ONNX file')
    args = parser.parse_args()
    zip_path = args.zip_path
    output_dir = args.output_dir
    logs_path = join(output_dir, 'logs.json')

    # Initialize the logs object
    logs = Logs()

    # Save the resulting Hailo-ONNX file with a new name
    makedirs(output_dir, exist_ok=True)

    try:
        # Unzip the file
        onnx_path, input_js, calibration_images = unzip_file(zip_path, join(output_dir, 'tmp'))

        onnx_filename = splitext(split(onnx_path)[1])[0]
        new_onnx_path = join(output_dir, onnx_filename) + '.hailo8.onnx'

        # Convert to Hailo ONNX
        logs.add_message('Converting ONNX to Hailo-ONNX', {'Input Path': onnx_path})
        convert_to_hailo_onnx(onnx_path, new_onnx_path, input_js, calibration_images, logs)
        # Create a valid mimetype
        mime_type = 'application/x-onnx; device=hailo-8'
        logs.add_message('Successful Conversion', {'MIME type': mime_type,
                                                   'Output Path': new_onnx_path})
    except Exception as e:
        logs.add_message('Conversion Failed',
                         {'Human Error': "The ONNX model is not supported by the Hailo SDK.",
                          'Technical Error': str(e)})

    logs.add_data(**{'Logs path': logs_path})

    logs.save_as_json(logs_path)
    print('Exiting...')
