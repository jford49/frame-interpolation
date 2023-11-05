r"""A test script for mid frame layer generation.

Usage example:
 python3 -m frame_interpolation.eval.interpolator_test \
   --folder_in '<folderpath of the source images>' \
   --n_layers <number of layers to create> \
   --model_path <The filepath of the TF2 saved model to use>
"""
import os
from os import path, listdir
from datetime import datetime
from typing import Sequence

from . import interpolator as interpolator_lib
from . import util
from absl import app
from absl import flags
import numpy as np

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


_FOLDER_IN = flags.DEFINE_string(
    name='folder_in',
    default=None,
    help='The folder with source images.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_N_LAYERS = flags.DEFINE_integer(
    name='n_layers',
    default=1,
    help='The number of layers to create.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=64,
    help='If >1, pad the input size so it is evenly divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='An int >= 1, number of patches along height, '
    'patch_height = height//block_height, should be evenly divisible.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='An int >= 1, number of patches along width, '
    'patch_width = width//block_width, should be evenly divisible.')

def _run_interpolator() -> None:
  """Writes interpolated mid frames from a given folder."""

  interpolator = interpolator_lib.Interpolator(
      model_path=_MODEL_PATH.value,
      align=_ALIGN.value,
      block_shape=[_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
  
  # Batched time.
  batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

  folder_in = _FOLDER_IN.value
  for iLayer in range(_N_LAYERS.value):
    now = datetime.now()
    dt_string = now.strftime("%H:%M:%S")
    print("\nlayer", str(iLayer+1)+"/"+str_N_LAYERS.value), "start: ", dt_string)

    folder_out = path.join(_FOLDER_IN.value, "L"+str(iLayer+1))
    if not path.isdir(folder_out):
      os.mkdir(folder_out)
  
    image_path_list = [f for f in listdir(folder_in) if path.isfile(path.join(folder_in, f))]
    image_path_list.sort()
  
    img_idx = 0
    if iLayer % 2 == 1:
      # Preserve the first image so that the input and output frame counts are the same
      photo1_path = path.join(folder_in, image_path_list[0])
      image_1 = util.read_image(photo1_path)
      first_frame_filepath = path.join(folder_out,"img"+f"{0:05d}"+".png")
      util.write_image(first_frame_filepath, image_1)
      img_idx += 1
  
    idx = 1
    while idx < len(image_path_list):
      photo1_path = path.join(folder_in, image_path_list[idx-1])
      photo2_path = path.join(folder_in, image_path_list[idx])
      
      image_1 = util.read_image(photo1_path)
      image_batch_1 = np.expand_dims(image_1, axis=0)
      image_2 = util.read_image(photo2_path)
      image_batch_2 = np.expand_dims(image_2, axis=0)
      
      # Invoke the model for one mid-frame interpolation.
      mid_frame = interpolator(image_batch_1, image_batch_2, batch_dt)[0]
      
      mid_frame_filepath = path.join(folder_out,"img"+f"{img_idx:05d}"+".png")
      util.write_image(mid_frame_filepath, mid_frame)
      idx+=1
      img_idx+=1
    
    if iLayer % 2 == 0:
      # Preserve the last image so that the input and output frame counts are the same
      photo1_path = path.join(folder_in, image_path_list[len(image_path_list)-1])
      image_1 = util.read_image(photo1_path)
      first_frame_filepath = path.join(folder_out,"img"+f"{img_idx:05d}"+".png")
      util.write_image(first_frame_filepath, image_1)

    folder_in = folder_out

def main(argv: Sequence[str]) -> None:
  #print(len(argv), "arguments")
  #for i in range(len(argv)):
  #  print(i, argv[i])
   
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments. Try quoting path names.')
    
  _run_interpolator()

if __name__ == '__main__':
  app.run(main)
