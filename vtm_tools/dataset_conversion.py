# Dataset Conversion for MPEG VCM Anchor Generation with Super Resolution
#
# by Joungil Yun (sigipus@etri.re.kr)

import os
import sys
import time
import multiprocessing
import subprocess
import argparse
import parmap
import numpy as np
import torch
import glob

from PIL import Image
import skimage.io as io
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

import argparse


def create_path(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("[Error] Cannot create a path: {}".format(path))


def yuv_format(chroma, bit):
    ret = 'yuv{}p'.format(chroma)
    if bit == 10:
        ret = ret + '10le'
    return ret


def conversion_function(params):
    index = params[0]
    command_list = params[1]

    for command_string in command_list:
        subprocess.run(command_string, shell=True)
        # popen = subprocess.Popen(command_string, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # (stdoutdata, stderrdata) = popen.communicate()
        # if stderrdata.decode('utf-8') != '':
        #     print(stdoutdata.decode('utf-8'), stderrdata.decode('utf-8'))
        #     raise Exception('[Error] Running command: {}'.format(command_string))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None, \
        help='input image file directory')
    parser.add_argument('--converted_dir', type=str, default=None, \
        help='converted image file directory')
    parser.add_argument('--bitstream_dir', type=str, default=None, \
        help='encoded bitstream file director')
    parser.add_argument('--input_file', type=str, default=None, \
        help='input file that contains a list of image file names')
    # parser.add_argument('--vtm_ver', type=str, default='8.2', \
    #     help='VTM version (8.2 or 12.0)')
    parser.add_argument('--encoder', type=str, default='vtm_12.0', \
        help='encoder')
    parser.add_argument('--bin_dir', type=str, default='./bin', \
        help='bin director')
    parser.add_argument('--qp', type=int, default=22, \
        help='encoding qp (22, 27, 32, 37, 42, 47)')
    parser.add_argument('--ds_level', type=int, default=0, \
        help='down-sacling level (0:100%, 1:75%, 2:50%, 3:25%)')
    parser.add_argument('--cq_timing', type=str, default='after_yuv',
        help='timing of color quantization')    
    parser.add_argument('--cq_method', type=str, default='median_cut',
        help='an algorithm used in color quantization')
    parser.add_argument('--cq_colors', type=int, default=256,
        help='a number of colors on color palette')

    args = parser.parse_args()

    if args.input_dir == None or args.converted_dir == None or args.bitstream_dir == None: 
        assert False, print("Input/Converted/Bitstream directory should be given")


    num_cores = multiprocessing.cpu_count()-1

    paramVTM_Level = 4
    paramVTM_chroma_format = 420
    paramVTM_InputBitDepth = 8
    paramVTM_OutputBitDepth = 10

    commandFFMPEG_base = '{}/ffmpeg -y -loglevel error'.format(args.bin_dir)

    if args.encoder == 'vtm_8.2':
        commandVTM_base = '{}/EncoderApp_8_2 -c ./configs/VTM/encoder_intra_vtm_8_2.cfg'.format(args.bin_dir)
    elif args.encoder == 'vtm_12.0':
        commandVTM_base = '{}/EncoderApp_12_0 -c ./configs/VTM/encoder_intra_vtm_12_0.cfg'.format(args.bin_dir)
    elif args.encoder == 'jpeg':
        commandJPEG_base = '{}/ffmpeg '
    else:    
        assert False, print("Unsupported VTM version")


    if args.qp in [22, 27, 32, 37, 42, 47]:
        paramVTM_QP = args.qp
    else:
        assert False, print("Unsupported QP")


    if args.ds_level in [0, 1, 2, 3]:
        ds_level = args.ds_level
    else:
        assert False, print("Unsupported DS")
    scale = 100 - 25 * ds_level


    input_list = None
    if args.input_file is not None:
        with open(args.input_file, 'r') as f:
            input_list = [x for x in f.read().splitlines()] #remove .png

    input_dir = os.path.abspath(args.input_dir)
    converted_dir = os.path.abspath(args.converted_dir)
    bitstream_dir = os.path.abspath(args.bitstream_dir)



    start_time = time.time()

    vtm_input_yuv_format = yuv_format(paramVTM_chroma_format, paramVTM_InputBitDepth)
    vtm_output_yuv_format = yuv_format(paramVTM_chroma_format, paramVTM_OutputBitDepth)
    output_image_format = 'rgb24'



    if ds_level == 0:
        commandFFMPEG_image_down_scaling_template = ' '.join(
            [
                commandFFMPEG_base,
                '-i {} -vf "pad={}:{}" {}'
            ]
        )
    else:
        commandFFMPEG_image_down_scaling_template = ' '.join(
            [
                commandFFMPEG_base,
                '-i {} -vf "scale={}:{}" {}'
            ]
        )


    commandFFMPEG_image_to_yuv_template = ' '.join(
        [
            commandFFMPEG_base,
            '-i {} -f rawvideo',
            '-pix_fmt {}'.format(vtm_input_yuv_format),
            '-dst_range 1 {}'
        ]
    )

    if 'vtm' in args.encoder:
        commandVTM_encoding_template = ' '.join(
            [
                commandVTM_base,
                #'--InputFile={} --SourceWidth={} --SourceHeight={}',
                #'--InputBitDepth={}'.format(paramVTM_InputBitDepth),
                #'--QP={}'.format(paramVTM_QP),
                #'--BitstreamFile={} --ReconFile={}',
                #'--OutputBitDepth={}'.format(paramVTM_OutputBitDepth),
                #'--FrameRate=1 --FramesToBeEncoded=1 --InternalBitDepth=10',
                '-i {} -o {} -b {} -q {}',
                '--ConformanceWindowMode=1 -wdt {} -hgt {}',
                '-f 1 -fr 1 --InternalBitDepth=10',
                '> {}'
            ]
        )
    else:
        commandJPEG_encoding_template = ' '.join(
            [
                commandFFMPEG_base,
                '-f rawvideo -s {s} -pix_fmt {pix_fmt} -i {i} -q:v {qv} {j}',
                '&&',
                commandFFMPEG_base,
                '-i {j} -s {s} -pix_fmt {pix_fmt} {r}',
                '&&',
                'mv {j} {b}'
            ]
        )

    commandFFMPEG_yuv_to_image_template = ' '.join(
        [
            commandFFMPEG_base,
            '-f rawvideo -pix_fmt {}',
            '-s {}',
            '-src_range 1 -i {} -frames 1',
            '-pix_fmt {}'.format(output_image_format),
            '{}'
        ]
    )

    if ds_level == 0:
        commandFFMPEG_image_up_scaling_template = ' '.join(
            [
                commandFFMPEG_base,
                '-i {} -vf "crop={}:{}:0:0" {}'
            ]
        )
    else:
        commandFFMPEG_image_up_scaling_template = ' '.join(
            [
                commandFFMPEG_base,
                '-i {} -vf "scale={}:{}" {}'
            ]
        )

    color_quantization_template = ('python ./src/OpenImages/color_quantize.py'
            ' --input_file={} --colors={} --method={} --output_file={}')

    print('Converting Files for QP{:02d}_{:03d}'.format(paramVTM_QP, scale))

    converted_path = os.path.join(converted_dir, 'QP{:02d}_{:03d}'.format(paramVTM_QP, scale))
    create_path(converted_path)

    bitstream_path = os.path.join(bitstream_dir, 'QP{:02d}_{:03d}'.format(paramVTM_QP, scale))
    create_path(bitstream_path)

    temp_file_path = os.path.join(converted_dir, 'temp')
    create_path(temp_file_path)


    file_list_all = [(path, files) for path, dirs, files in os.walk(input_dir)]
    file_list_all.sort()

    file_list = {path: [file for file in files if
                        file.endswith('.png') or file.endswith('.jpg') or
                        file.endswith('.PNG') or file.endswith('.JPG')]
                for path, files in file_list_all}

    for key in file_list.keys():
        file_list[key].sort()

    if input_list is not None:
        for file_path in file_list.keys():
            temp_list = file_list[file_path].copy()
            for file in temp_list:
                if file not in input_list:
                    file_list[file_path].remove(file)

    params_list = []

    for file_path in file_list.keys():


        for img_idx, file in enumerate(file_list[file_path]):
            file_name = os.path.splitext(os.path.basename(file))[0]
            # print('{:04d} : {}'.format(img_idx, file_name))

            input_image_file = os.path.join(file_path, file)
            img = Image.open(input_image_file)


            width_original, height_original = img.size

            if ds_level == 0:
                width_down = int(np.ceil(width_original/2)*2)
                height_down = int(np.ceil(height_original/2)*2)
            else:
                width_down = int(np.ceil(width_original * (4 - ds_level) / 8) * 2)
                height_down = int(np.ceil(height_original * (4 - ds_level) / 8) * 2)

            resolution_down = '{}x{}'.format(width_down, height_down)


            # 1. <FFMPEG> Down-Scaling
            down_scaled_temp_file = os.path.join(temp_file_path, 'downscaled_' + file_name + '.png')
            commandFFMPEG_image_down_scaling = commandFFMPEG_image_down_scaling_template.format(
                input_image_file,
                width_down,
                height_down,
                down_scaled_temp_file
            )

            # (Before) Color Quantization
            if args.cq_timing == 'before_yuv':
                command_color_quantization = color_quantization_template.format(
                    down_scaled_temp_file, args.cq_colors, args.cq_method, down_scaled_temp_file
                )

            # 2. <FFMPEG> Image2YUV
            image_to_yuv_temp_file = os.path.join(
                temp_file_path,
                file_name + '_' + resolution_down + '_' + vtm_input_yuv_format + '.yuv'
            )
            commandFFMPEG_image_to_yuv = commandFFMPEG_image_to_yuv_template.format(
                down_scaled_temp_file,
                image_to_yuv_temp_file)

            # (After) Color Quantization
            if args.cq_timing == 'after_yuv':
                command_color_quantization = color_quantization_template.format(
                    image_to_yuv_temp_file, args.cq_colors, args.cq_method, image_to_yuv_temp_file
                )

            # 3. <VTM> Encoding (YUV2YUV) w/ QP
            bitstream_file_path = file_path.replace(input_dir, bitstream_path, 1)
            create_path(bitstream_file_path)
            if 'vtm' in args.encoder:
                InputFile = image_to_yuv_temp_file
                paramVTM_SourceWidth = width_down
                paramVTM_SourceHeight = height_down

                BitstreamFile = os.path.join(bitstream_file_path, file_name + '.bin')
                ReconFile = os.path.join(temp_file_path,
                                                file_name + '_vtm_recon_' + resolution_down + '_' + vtm_output_yuv_format + '.yuv')
                vtm_log_file = os.path.join(temp_file_path, file_name + '.log')
                command_encoding = commandVTM_encoding_template.format(
                    InputFile,
                    ReconFile,
                    BitstreamFile,
                    paramVTM_QP,
                    paramVTM_SourceWidth,
                    paramVTM_SourceHeight,
                    vtm_log_file
                )
            else:
                JPEGFile = os.path.join(bitstream_file_path, file_name + '.jpg')
                BitstreamFile = os.path.join(bitstream_file_path, file_name + '.bin')
                ReconFile = os.path.join(temp_file_path,
                                                file_name + '_jpeg_recon_' + resolution_down + '_' + vtm_input_yuv_format + '.yuv')
                command_encoding = commandJPEG_encoding_template.format(
                    i=image_to_yuv_temp_file,
                    s=resolution_down,
                    pix_fmt=vtm_input_yuv_format,
                    j=JPEGFile,
                    r=ReconFile,
                    b=BitstreamFile,
                    qv=8,
                )


            # 4. <FFMPEG> YUV2Image
            yuv_to_image_temp_file = os.path.join(temp_file_path, 'decoded_' + file_name + '.png')
            commandFFMPEG_yuv_to_image = commandFFMPEG_yuv_to_image_template.format(
                vtm_output_yuv_format if 'vtm' in args.encoder else vtm_input_yuv_format,
                resolution_down,
                ReconFile,
                yuv_to_image_temp_file
            )

            # 5. <FFMPEG> or <SR> Up-Scaling
            converted_file_path = file_path.replace(input_dir, converted_path, 1)
            create_path(converted_file_path)             
            output_image_file = os.path.join(converted_file_path, file_name + '.png')
            commandFFMPEG_image_up_scaling = commandFFMPEG_image_up_scaling_template.format(
                yuv_to_image_temp_file,
                width_original,
                height_original,
                output_image_file
            )

            command_list = []
            # Pay attention to the order of adding command string to the list
            command_list.append(commandFFMPEG_image_down_scaling)
            if args.cq_timing == 'before_yuv' and args.cq_method != 'none':
                command_list.append(command_color_quantization)  # color quantization
            command_list.append(commandFFMPEG_image_to_yuv)
            if args.cq_timing == 'after_yuv' and args.cq_method != 'none':
                command_list.append(command_color_quantization)  # color quantization
            command_list.append(command_encoding)
            command_list.append(commandFFMPEG_yuv_to_image)
            command_list.append(commandFFMPEG_image_up_scaling)

            params_list.append([img_idx, command_list])

    result = parmap.map(conversion_function, params_list, pm_processes=os.cpu_count(), pm_chunksize=1, pm_pbar=True)

    end_time = time.time()
    print('The processing time is {}(sec).\n'.format(end_time-start_time))

