from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import onnxruntime
from dfl.xlib.image import ImageProcessor


def get_dims(img: np.ndarray) -> tuple:
    """Returns the dimensions of the image (N, H, W, C)."""
    if img.ndim == 2:
        # Grayscale image (H, W)
        return 1, *img.shape, 1
    elif img.ndim == 3:
        # RGB or Grayscale with single channel (H, W, C)
        return 1, *img.shape
    elif img.ndim == 4:
        # Batch dimension is present (N, H, W, C)
        return img.shape
    else:
        raise ValueError("Unsupported image dimension")

def get_dtype(img: np.ndarray) -> np.dtype:
    """Returns the data type of the image."""
    return img.dtype

def resize_image(img: np.ndarray, size: tuple) -> np.ndarray:
    """Resizes the image to the given size."""
    pil_img = Image.fromarray(img.astype('uint8'))
    resized_img = pil_img.resize(size, Image.ANTIALIAS)
    return np.array(resized_img)

def change_channels(img: np.ndarray, num_channels: int) -> np.ndarray:
    """Changes the number of channels of the image."""
    if num_channels == 3 and img.ndim == 2:
        # Convert grayscale to RGB
        return np.stack([img] * 3, axis=-1)
    elif num_channels == 1 and img.ndim == 3:
        # Convert RGB to grayscale
        return np.mean(img, axis=-1, keepdims=True)
    elif img.ndim == 3 and img.shape[-1] != num_channels:
        raise ValueError(f"Image already has a different number of channels: {img.shape[-1]}")
    return img

def convert_to_float32(img: np.ndarray) -> np.ndarray:
    """Converts the image to float32 format with values in [0, 1]."""
    return img.astype(np.float32) / 255.0

def rearrange_image(img: np.ndarray, format: str) -> np.ndarray:
    """Returns the image array in the desired format."""
    format = format.upper()
    N_slice = 0 if 'N' not in format else slice(None)
    H_slice = 0 if 'H' not in format else slice(None)
    W_slice = 0 if 'W' not in format else slice(None)
    C_slice = 0 if 'C' not in format else slice(None)
    img = img[N_slice, H_slice, W_slice, C_slice]

    # Current format string construction
    current_format = ''
    if 'N' in format: current_format += 'N'
    if 'H' in format: current_format += 'H'
    if 'W' in format: current_format += 'W'
    if 'C' in format: current_format += 'C'

    if current_format != format:
        # Create a mapping from current format to its index positions
        format_mapping = {dim: i for i, dim in enumerate(current_format)}
        # Determine the transpose order to achieve the desired format
        transpose_order = [format_mapping[dim] for dim in format]
        img = img.transpose(transpose_order)

    return np.ascontiguousarray(img)

class DFMModel:
    def __init__(self, model_path: str, providers, device=None):

        self._model_path = model_path
        self.providers = providers
        sess = self._sess = onnxruntime.InferenceSession(str(model_path), providers=self.providers)
        inputs = sess.get_inputs()

        if len(inputs) == 0 or 'in_face' not in inputs[0].name:
            raise Exception(f'Invalid model {model_path}')
        
        self._input_height, self._input_width = inputs[0].shape[1:3]
        self._model_type = 1
        
        if len(inputs) == 2:
            if 'morph_value' not in inputs[1].name:
                raise Exception(f'Invalid model {model_path}')
            self._model_type = 2
        elif len(inputs) > 2:
            raise Exception(f'Invalid model {model_path}')

    def get_model_path(self) -> Path: 
        return self._model_path

    def get_input_res(self) -> Tuple[int, int]:
        return self._input_width, self._input_height

    def has_morph_value(self) -> bool:
        return self._model_type == 2

    def convert(self, img, morph_factor=0.75, rct=False):
        """
         img    np.ndarray  HW,HWC,NHWC uint8,float32

         morph_factor   float   used if model supports it

        returns

         img        NHW3  same dtype as img
         celeb_mask NHW1  same dtype as img
         face_mask  NHW1  same dtype as img
        """

        img = img[:, :, ::-1]
        # img = np.rot90(img, k=3)


        ip = ImageProcessor(img)

        N,H,W,C = ip.get_dims()
        dtype = ip.get_dtype()

        img = ip.resize( (self._input_width,self._input_height) ).ch(3).to_ufloat32().get_image('NHWC')


        if self._model_type == 1:
            out_face_mask, out_celeb, out_celeb_mask = self._sess.run(None, {'in_face:0': img})
        elif self._model_type == 2:
            out_face_mask, out_celeb, out_celeb_mask = self._sess.run(None, {'in_face:0': img, 'morph_value:0':np.float32([morph_factor]) })

        out_celeb      = ImageProcessor(out_celeb).resize((W,H)).ch(3).to_dtype(dtype).get_image('NHWC')
        out_celeb_mask = ImageProcessor(out_celeb_mask).resize((W,H)).ch(1).to_dtype(dtype).get_image('NHWC')
        out_face_mask  = ImageProcessor(out_face_mask).resize((W,H)).ch(1).to_dtype(dtype).get_image('NHWC')


        out_face_mask = ImageProcessor(out_face_mask).get_image('HWC')
        out_face_mask = out_face_mask[:, :, ::-1]

        if rct:
            out_celeb = ImageProcessor(out_celeb).rct(ImageProcessor(img).resize((W,H)).get_image('NHWC'), out_celeb_mask, out_celeb_mask, 0.3)
            out_celeb = out_celeb.get_image('HWC')
            out_celeb = out_celeb[:, :, ::-1]
        # plt.imshow(out_celeb_rct)
        # plt.show()
        else:
            out_celeb = ImageProcessor(out_celeb).get_image('HWC')
            out_celeb = out_celeb[:, :, ::-1]
        # plt.imshow(out_celeb)
        # plt.show()

        out_celeb_mask = ImageProcessor(out_celeb_mask).get_image('HWC')
        out_celeb_mask = out_celeb_mask[:, :, ::-1]

        
        return out_celeb, out_celeb_mask, out_face_mask
    
    def get_fai_ip(self, img):
        fai_ip = ImageProcessor(img)
        return fai_ip
