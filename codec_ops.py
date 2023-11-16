import math
import tempfile
import subprocess

from pathlib import Path
from PIL import Image
from io import BytesIO

import ray
import numpy as np


def codec_fn(x, codec, quality):
    assert codec in ['jpeg', 'webp', 'vtm']

    x = x.copy()
    x *= 255.
    x = x.astype('uint8')
    x = x.transpose((1, 2, 0))
    pixels = x.shape[0] * x.shape[1]

    pil_img = Image.fromarray(x)

    if codec != 'vtm':
        mem = BytesIO()
        pil_img.save(mem, codec, quality=quality, subsampling=0)
        pil_img_recon = Image.open(mem)
        bpp = mem.getbuffer().nbytes * 8 / pixels
    else:
        pil_img_recon, bpp = run_vtm(pil_img, qp=quality)

    x = np.asarray(pil_img_recon)
    pil_img.close()
    pil_img_recon.close()
    if codec != 'vtm':
        mem.close()

    x = x.astype('float32')
    x = x.transpose((2, 0, 1))
    x /= 255.
    return x, bpp


def run_vtm(pil_img, qp, ds_level=0, tool_path='/surrogate/vtm_tools'):
    dst_dir_obj = tempfile.TemporaryDirectory()
    dst_dir = Path(dst_dir_obj.name)

    w, h = pil_img.size
    dw, dh = map(lambda x: math.ceil(x / 2) * 2, [w, h])

    src_img_path = str(dst_dir / 'raw.png')
    pil_img.save(src_img_path)
    file_name = 'img'

    tmp_path = dst_dir / (file_name + '_tmp.png')
    yuv_path = dst_dir / (file_name + '.yuv')
    vtm_recon_yuv_path = dst_dir / (file_name + '_vtm_recon.yuv')
    vtm_bin_path = dst_dir / (file_name + '_vtm.bin')
    log_path = dst_dir / (file_name + '.log')
    vtm_recon_png_path = dst_dir / (file_name + '_vtm_recon.png')

    ffmpeg_base_cmd = f"{tool_path}/bin/ffmpeg -y -loglevel error"

    if ds_level == 0:
        down_cmd = (ffmpeg_base_cmd + " -i {} -vf 'pad={}:{}' {}").format(src_img_path, dw, dh, tmp_path)
        up_cmd = (ffmpeg_base_cmd + " -i {} -vf 'crop={}:{}:0:0' {}").format(tmp_path, w, h, vtm_recon_png_path)
    else:
        down_cmd = (ffmpeg_base_cmd + " -i {} -vf 'scale={}:{}' {}").format(src_img_path, dw, dh, tmp_path)
        up_cmd = (ffmpeg_base_cmd + " -i {} -vf 'scale={}:{}' {}").format(tmp_path, w, h, vtm_recon_png_path)
        
    img2yuv_cmd = ffmpeg_base_cmd + " -i {} -f rawvideo -pix_fmt yuv420p -dst_range 1 {}".format(tmp_path, yuv_path)
    vtm_cmd = (f"{tool_path}/bin/EncoderApp_12_0 -c {tool_path}/configs/VTM/encoder_intra_vtm_12_0.cfg "
               "-i {} -o {} -b {} -q {} --ConformanceWindowMode=1 -wdt {} -hgt {} "
               "-f 1 -fr 1 --InternalBitDepth=10 > {}").format(yuv_path, vtm_recon_yuv_path, vtm_bin_path, qp, dw, dh, log_path)
    yuv2img_cmd = (ffmpeg_base_cmd + " -f rawvideo -pix_fmt yuv420p10le -s {} -src_range 1 -i {} -frames 1 "
                   "-pix_fmt rgb24 {}").format(f'{dw}x{dh}', vtm_recon_yuv_path, tmp_path)

    subprocess.run(down_cmd, shell=True)
    subprocess.run(img2yuv_cmd, shell=True)
    subprocess.run(vtm_cmd, shell=True)
    subprocess.run(yuv2img_cmd, shell=True)
    subprocess.run(up_cmd, shell=True)

    recon_img = Image.open(vtm_recon_png_path)
    bpp = Path(vtm_bin_path).stat().st_size * 8 / (h * w)
    return recon_img, bpp


@ray.remote
def ray_codec_fn(x, codec, quality):
    return codec_fn(x, codec, quality)


@ray.remote
def ray_put_codec_fn(xs, codec, quality, idx):
    return codec_fn(xs[idx], codec, quality)


def batch_codec_fn(xs, codec, quality, use_put=True):
    if use_put:
        n = len(xs)
        xs = ray.put(xs)
        outs = ray.get([ray_put_codec_fn.remote(xs, codec, quality, i) for i in range(n)])
        xs, bpps = zip(*outs)
    else:
        outs = ray.get([ray_codec_fn.remote(x, codec, quality) for x in xs])
        xs, bpps = zip(*outs)
    xs = np.stack(xs, axis=0)
    bpps = np.array(bpps, dtype='float32')
    return {'x_hat': xs, 'bpp': bpps}