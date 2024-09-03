import cv2
import numpy as np
from skimage import transform as trans
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from numpy.linalg import norm as l2norm
import onnxruntime
import onnx
from itertools import product as product
import subprocess as sp
onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1
import rope.FaceUtil as faceutil
import pickle
import math
from torch.cuda import nvtx
from collections import OrderedDict
import platform
from rope.EngineBuilder import onnx_to_trt as onnx2trt
import os
from packaging import version
from dfl.DFMModel import DFMModel
from dfl.xlib.onnxruntime.device import ORTDeviceInfo
from queue import Queue
from threading import Lock

try:
    import tensorrt as trt
    import ctypes
except ModuleNotFoundError:
    print("No TensorRT Found")

# Dizionario per la conversione dei tipi di dati numpy a torch
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Usa lo stesso logger globale di TensorRT come nella classe EngineBuilder
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# imported from https://github.com/warmshao/FasterLivePortrait/blob/master/src/models/predictor.py
# adjusted to work with TensorRT 10.3.0
class TensorRTPredictor:
    """
    Implements inference for the TensorRT engine with a pool of execution contexts.
    """

    def __init__(self, **kwargs):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        :param pool_size: The size of the pool of execution contexts.
        """
        global TRT_LOGGER

        # Inizializzazione del modello TensorRT
        self.engine = None
        self.context_pool = None
        self.lock = Lock()

        custom_plugin_path = kwargs.get("custom_plugin_path", None)
        if custom_plugin_path is not None:
            # Carica il plugin personalizzato solo una volta
            if platform.system().lower() == 'linux':
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
            else:
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)

        # Load TRT engine
        engine_path = kwargs.get("model_path", None)
        self.debug = kwargs.get("debug", False)
        self.pool_size = kwargs.get("pool_size", 10)
        assert engine_path, f"model:{engine_path} must exist!"

        # Caricamento dell'engine TensorRT
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine

        # Setup I/O bindings e contesto
        self.inputs = []
        self.outputs = []
        self.tensors = OrderedDict()

        # Gestione dei tensori dinamici
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            binding = {
                "index": idx,
                "name": name,
                "dtype": dtype,
                "shape": list(shape)
            }
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        self.allocate_max_buffers()

        # Creazione del pool di contesti di esecuzione
        self.context_pool = Queue(maxsize=self.pool_size)
        for _ in range(self.pool_size):
            self.context_pool.put(self.engine.create_execution_context())

    def allocate_max_buffers(self, device="cuda"):
        nvtx.range_push("allocate_max_buffers")
        # Supporto per batch dinamico
        batch_size = 1
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine.get_tensor_name(idx)
            shape = self.engine.get_tensor_shape(binding)
            is_input = self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT
            if -1 in shape:
                if is_input:
                    shape = self.engine.get_tensor_profile_shape(binding, 0)[-1]
                    batch_size = shape[0]
                else:
                    shape[0] = batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor
        nvtx.range_pop()

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        specs = []
        for i, o in enumerate(self.inputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt input {i} -> {o['name']} -> {o['shape']}")
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for i, o in enumerate(self.outputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt output {i} -> {o['name']} -> {o['shape']}")
        return specs

    def adjust_buffer(self, feed_dict, context):
        """
        Adjust input buffer sizes and set input shapes in the given execution context.
        :param feed_dict: A dictionary of inputs as numpy arrays.
        :param context: The TensorRT execution context to set input shapes.
        """
        nvtx.range_push("adjust_buffer")
        for name, buf in feed_dict.items():
            input_tensor = self.tensors[name]
            current_shape = list(buf.shape)
            slices = tuple(slice(0, dim) for dim in current_shape)
            input_tensor[slices].copy_(buf)
            # Imposta la forma di input nel contesto fornito
            context.set_input_shape(name, current_shape)
        nvtx.range_pop()

    def predict(self, feed_dict):
        """
        Execute inference on a batch of images in synchronous mode using execute_v2.
        :param feed_dict: A dictionary of inputs as numpy arrays.
        :return: A dictionary of outputs as PyTorch tensors.
        """
        # Ottieni un contesto dal pool
        with self.lock:
            context = self.context_pool.get()

        try:
            nvtx.range_push("set_tensors")
            # Passa il contesto a adjust_buffer
            self.adjust_buffer(feed_dict, context)

            for name, tensor in self.tensors.items():
                assert tensor.dtype == torch.float32, f"Tensor '{name}' should be torch.float32 but is {tensor.dtype}"
                context.set_tensor_address(name, tensor.data_ptr())
            
            nvtx.range_pop()

            # Prepara i binding per execute_v2()
            bindings = [tensor.data_ptr() for tensor in self.tensors.values()]

            # Esecuzione sincrona con execute_v2()
            nvtx.range_push("execute")
            noerror = context.execute_v2(bindings)
            if not noerror:
                raise ValueError("ERROR: inference failed.")
            nvtx.range_pop()

            return self.tensors

        finally:
            # Restituisci il contesto al pool dopo l'uso
            with self.lock:
                self.context_pool.put(context)

    def predict_async(self, feed_dict, stream):
        """
        Execute inference on a batch of images in asynchronous mode using execute_async_v3.
        :param feed_dict: A dictionary of inputs as numpy arrays.
        :param stream: A CUDA stream for asynchronous execution.
        :return: A dictionary of outputs as PyTorch tensors.
        """
        # Ottieni un contesto dal pool
        with self.lock:
            context = self.context_pool.get()

        try:
            nvtx.range_push("set_tensors")
            # Passa il contesto a adjust_buffer
            self.adjust_buffer(feed_dict, context)

            for name, tensor in self.tensors.items():
                assert tensor.dtype == torch.float32, f"Tensor '{name}' should be torch.float32 but is {tensor.dtype}"
                context.set_tensor_address(name, tensor.data_ptr())
            
            nvtx.range_pop()

            # Esecuzione asincrona con execute_async_v3()
            nvtx.range_push("execute_async")
            noerror = context.execute_async_v3(stream)
            if not noerror:
                raise ValueError("ERROR: inference failed.")
            nvtx.range_pop()

            return self.tensors

        finally:
            # Restituisci il contesto al pool dopo l'uso
            with self.lock:
                self.context_pool.put(context)

    def cleanup(self):
        """
        Clean up all resources associated with the TensorRTPredictor.
        This method should be called explicitly before deleting the object.
        """
        # Pulisci l'engine TensorRT
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine  # Libera l'engine di TensorRT
            self.engine = None  # Imposta a None per assicurarti che il GC lo raccolga

        # Pulisci il pool di contesti di esecuzione
        if hasattr(self, 'context_pool') and self.context_pool is not None:
            while not self.context_pool.empty():
                context = self.context_pool.get()
                del context  # Libera ogni contesto
            self.context_pool = None  # Imposta a None per il GC

        # Imposta gli attributi su None per garantire la pulizia
        self.inputs = None
        self.outputs = None
        self.tensors = None
        self.pool_size = None

    def __del__(self):
        # Richiama il metodo cleanup nel distruttore per maggiore sicurezza
        self.cleanup()

class Models():
    def __init__(self):
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        #self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.providers = [
            ('CUDAExecutionProvider'),
            ('CPUExecutionProvider')]
        self.provider_name = 'CUDAExecutionProvider'
        self.retinaface_model = []
        self.yoloface_model = []
        self.scrdf_model = []
        self.yunet_model = []
        self.face_landmark_68_model = []
        self.face_landmark_3d68_model = []
        self.mean_lmk = []
        self.face_landmark_98_model = []
        self.face_landmark_106_model = []
        self.face_landmark_203_model = []
        self.face_landmark_478_model = []
        self.face_blendshapes_model = []
        self.resnet50_model, self.anchors  = [], []

        self.insight106_model = []

        self.recognition_model = []
        self.recognition_simswap_model = []
        self.recognition_ghost_model = []
        self.swapper_model = []
        self.simswap512_model = []
        self.ghostfacev1swap_model = []
        self.ghostfacev2swap_model = []
        self.ghostfacev3swap_model = []

        self.dfl_models = {}
        self.emap = []
        self.GFPGAN_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.GPEN_1024_model = []
        self.GPEN_2048_model = []
        self.codeformer_model = []
        self.VQFR_v2_model = []
        self.RestoreFormerPlusPlus_model = []
        self.realesrganx2plus_model = []
        self.realesrganx4plus_model = []
        self.realesrx4v3_model = []
        self.ultrasharpx4_model = []
        self.ultramixx4_model = []
        self.bsrganx2_model = []
        self.bsrganx4_model = []
        self.deoldify_art_model = []
        self.deoldify_stable_model = []
        self.deoldify_video_model = []
        self.ddcolor_art_model = []
        self.ddcolor_model = []

        self.occluder_model = []
        self.model_xseg = []
        self.faceparser_model = []

        self.syncvec = torch.empty((1,1), dtype=torch.float32, device='cuda:0')

        self.normalize = v2.Normalize(mean = [ 0., 0., 0. ],
                                      std = [ 1/1.0, 1/1.0, 1/1.0 ])

        self.LandmarksSubsetIdxs = [
            0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
            40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
            81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
            136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
            161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
            249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
            296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
            336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
            384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
            466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
        ]

        # Face Editor
        self.lp_motion_extractor_model = []
        self.lp_appearance_feature_extractor_model = []
        self.lp_stitching_model = []
        self.lp_stitching_eye_model = []
        self.lp_stitching_lip_model = []
        self.lp_warping_spade_fix_model = []
        self.lp_mask_crop = faceutil.create_faded_inner_mask(size=(512, 512), border_thickness=5, fade_thickness=15, blur_radius=5, device='cuda')
        self.lp_mask_crop = torch.unsqueeze(self.lp_mask_crop, 0)
        self.lp_mask_crop = torch.mul(self.lp_mask_crop, 255.)

    def switch_providers_priority(self, provider_name):
        match provider_name:
            case "TensorRT" | "TensorRT-Engine":
                providers = [
                                ('TensorrtExecutionProvider', {
                                    'trt_engine_cache_enable': True,
                                    'trt_engine_cache_path': "tensorrt-engines",
                                    'trt_timing_cache_enable': True,
                                    'trt_timing_cache_path': "tensorrt-engines",
                                    'trt_dump_ep_context_model': True,
                                    'trt_ep_context_file_path': "tensorrt-engines",
                                    'trt_layer_norm_fp32_fallback': True,
                                    'trt_builder_optimization_level': 5,
                                }),
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                if version.parse(trt.__version__) < version.parse("10.2.0") and provider_name == "TensorRT-Engine":
                    print("TensorRT-Engine provider cannot be used when TensorRT version is lower than 10.2.0.")
                    provider_name = "TensorRT"

            case "CPU":
                providers = [
                                ('CPUExecutionProvider'),
                                ('CUDAExecutionProvider')
                            ]
            case _:
                providers = [
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]

        self.providers = providers
        self.provider_name = provider_name

        return self.provider_name

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_total = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        memory_used = memory_total[0] - memory_free[0]

        return memory_used, memory_total[0]

    def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5, use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles:list[int]=[0]):
        bboxes = []
        kpss_5 = []
        kpss = []

        if detect_mode=='Retinaface':
            if not self.retinaface_model:
                self.retinaface_model = onnxruntime.InferenceSession('./models/det_10g.onnx', providers=self.providers)

            bboxes, kpss_5, kpss = self.detect_retinaface(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='SCRDF':
            if not self.scrdf_model:
                self.scrdf_model = onnxruntime.InferenceSession('./models/scrfd_2.5g_bnkps.onnx', providers=self.providers)

            bboxes, kpss_5, kpss = self.detect_scrdf(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='Yolov8':
            if not self.yoloface_model:
                self.yoloface_model = onnxruntime.InferenceSession('./models/yoloface_8n.onnx', providers=self.providers)

            bboxes, kpss_5, kpss = self.detect_yoloface(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='Yunet':
            if not self.yunet_model:
                self.yunet_model = onnxruntime.InferenceSession('./models/yunet_n_640_640.onnx', providers=self.providers)

            bboxes, kpss_5, kpss = self.detect_yunet(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        return bboxes, kpss_5, kpss

    def run_detect_landmark(self, img, bbox, det_kpss, detect_mode='203', score=0.5, from_points=False):
        kpss_5 = []
        kpss = []
        scores = []

        if detect_mode=='5':
            if not self.resnet50_model:
                self.resnet50_model = onnxruntime.InferenceSession("./models/res50.onnx", providers=self.providers)

                feature_maps = [[64, 64], [32, 32], [16, 16]]
                min_sizes = [[16, 32], [64, 128], [256, 512]]
                steps = [8, 16, 32]
                image_size = 512
                # re-initialize self.anchors due to clear_mem function
                self.anchors  = []

                for k, f in enumerate(feature_maps):
                    min_size_array = min_sizes[k]
                    for i, j in product(range(f[0]), range(f[1])):
                        for min_size in min_size_array:
                            s_kx = min_size / image_size
                            s_ky = min_size / image_size
                            dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                            dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                            for cy, cx in product(dense_cy, dense_cx):
                                self.anchors += [cx, cy, s_kx, s_ky]

            kpss_5, kpss, scores = self.detect_face_landmark_5(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

        elif detect_mode=='68':
            if not self.face_landmark_68_model:
                self.face_landmark_68_model = onnxruntime.InferenceSession('./models/2dfan4.onnx', providers=self.providers)

            kpss_5, kpss, scores = self.detect_face_landmark_68(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

        elif detect_mode=='3d68':
            if not self.face_landmark_3d68_model:
                self.face_landmark_3d68_model = onnxruntime.InferenceSession('./models/1k3d68.onnx', providers=self.providers)
                with open('./models/meanshape_68.pkl', 'rb') as f:
                    self.mean_lmk = pickle.load(f)

            kpss_5, kpss, scores = self.detect_face_landmark_3d68(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        elif detect_mode=='98':
            if not self.face_landmark_98_model:
                self.face_landmark_98_model = onnxruntime.InferenceSession('./models/peppapig_teacher_Nx3x256x256.onnx', providers=self.providers)

            kpss_5, kpss, scores = self.detect_face_landmark_98(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

        elif detect_mode=='106':
            if not self.face_landmark_106_model:
                self.face_landmark_106_model = onnxruntime.InferenceSession('./models/2d106det.onnx', providers=self.providers)

            kpss_5, kpss, scores = self.detect_face_landmark_106(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        elif detect_mode=='203':
            if not self.face_landmark_203_model:
                self.face_landmark_203_model = onnxruntime.InferenceSession('./models/landmark.onnx', providers=self.providers)

            kpss_5, kpss, scores = self.detect_face_landmark_203(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        elif detect_mode=='478':
            if not self.face_landmark_478_model:
                self.face_landmark_478_model = onnxruntime.InferenceSession('./models/face_landmarks_detector_Nx3x256x256.onnx', providers=self.providers)

            if not self.face_blendshapes_model:
                self.face_blendshapes_model = onnxruntime.InferenceSession('./models/face_blendshapes_Nx146x2.onnx', providers=self.providers)

            kpss_5, kpss, scores = self.detect_face_landmark_478(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        if len(kpss_5) > 0:
            if len(scores) > 0:
                if np.mean(scores) >= score:
                    return kpss_5, kpss, scores
            else:
                return kpss_5, kpss, scores

        return [], [], []

    def delete_models(self):
        self.retinaface_model = []
        self.yoloface_model = []
        self.scrdf_model = []
        self.yunet_model = []
        self.face_landmark_68_model = []
        self.face_landmark_3d68_model = []
        self.mean_lmk = []
        self.face_landmark_98_model = []
        self.face_landmark_106_model = []
        self.face_landmark_203_model = []
        self.face_landmark_478_model = []
        self.face_blendshapes_model = []
        self.resnet50_model = []
        self.insight106_model = []
        self.recognition_model = []
        self.recognition_simswap_model = []
        self.recognition_ghost_model = []
        self.swapper_model = []
        self.simswap512_model = []
        self.ghostfacev1swap_model = []
        self.ghostfacev2swap_model = []
        self.ghostfacev3swap_model = []
        self.GFPGAN_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.GPEN_1024_model = []
        self.GPEN_2048_model = []
        self.codeformer_model = []
        self.VQFR_v2_model = []
        self.RestoreFormerPlusPlus_model = []
        self.realesrganx2plus_model = []
        self.realesrganx4plus_model = []
        self.realesrx4v3_model = []
        self.ultramixx4_model = []
        self.ultrasharpx4_model = []
        self.bsrganx2_model = []
        self.bsrganx4_model = []
        self.deoldify_art_model = []
        self.deoldify_stable_model = []
        self.deoldify_video_model = []
        self.ddcolor_art_model = []
        self.ddcolor_model = []
        self.occluder_model = []
        self.model_xseg = []
        self.faceparser_model = []
        self.dfl_model = []
        self.dfl_models = {}

        # Face Editor
        if isinstance(self.lp_motion_extractor_model, TensorRTPredictor):
            # È un'istanza di TensorRTPredictor
            self.lp_motion_extractor_model.cleanup()

        del self.lp_motion_extractor_model
        self.lp_motion_extractor_model = None

        if isinstance(self.lp_appearance_feature_extractor_model, TensorRTPredictor):
            # È un'istanza di TensorRTPredictor
            self.lp_appearance_feature_extractor_model.cleanup()

        del self.lp_appearance_feature_extractor_model
        self.lp_appearance_feature_extractor_model = None

        if isinstance(self.lp_stitching_model, TensorRTPredictor):
            # È un'istanza di TensorRTPredictor
            self.lp_stitching_model.cleanup()

        del self.lp_stitching_model
        self.lp_stitching_model = None

        if isinstance(self.lp_stitching_eye_model, TensorRTPredictor):
            # È un'istanza di TensorRTPredictor
            self.lp_stitching_eye_model.cleanup()

        del self.lp_stitching_eye_model
        self.lp_stitching_eye_model = None

        if isinstance(self.lp_stitching_lip_model, TensorRTPredictor):
            # È un'istanza di TensorRTPredictor
            self.lp_stitching_lip_model.cleanup()

        del self.lp_stitching_lip_model
        self.lp_stitching_lip_model = None

        if isinstance(self.lp_warping_spade_fix_model, TensorRTPredictor):
            # È un'istanza di TensorRTPredictor
            self.lp_warping_spade_fix_model.cleanup()

        del self.lp_warping_spade_fix_model
        self.lp_warping_spade_fix_model = None

    def run_recognize(self, img, kps, similarity_type='Opal', face_swapper_model='Inswapper128'):
        if face_swapper_model == 'Inswapper128':
            if not self.recognition_model:
                self.recognition_model = onnxruntime.InferenceSession('./models/w600k_r50.onnx', providers=self.providers)

            embedding, cropped_image = self.recognize(self.recognition_model, img, kps, similarity_type=similarity_type)
        elif face_swapper_model == 'SimSwap512':
            if not self.recognition_simswap_model:
                self.recognition_simswap_model = onnxruntime.InferenceSession('./models/simswap_arcface_model.onnx', providers=self.providers)

            embedding, cropped_image = self.recognize(self.recognition_simswap_model, img, kps, similarity_type=similarity_type)

        elif face_swapper_model == 'GhostFace-v1' or face_swapper_model == 'GhostFace-v2' or face_swapper_model == 'GhostFace-v3':
            if not self.recognition_ghost_model:
                self.recognition_ghost_model = onnxruntime.InferenceSession('./models/ghost_arcface_backbone.onnx', providers=self.providers)

            embedding, cropped_image = self.recognize(self.recognition_ghost_model, img, kps, similarity_type=similarity_type)

        return embedding, cropped_image

    def calc_swapper_latent(self, source_embedding):
        if not self.swapper_model:
            graph = onnx.load("./models/inswapper_128.fp16.onnx").graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])

        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def run_swapper(self, image, embedding, output):
        if not self.swapper_model:
            cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False

            self.swapper_model = onnxruntime.InferenceSession( "./models/inswapper_128.fp16.onnx", providers=self.providers)

        io_binding = self.swapper_model.io_binding()
        io_binding.bind_input(name='target', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.swapper_model.run_with_iobinding(io_binding)

    def calc_swapper_latent_simswap512(self, source_embedding):
        latent = source_embedding.reshape(1, -1)
        latent /= np.linalg.norm(latent)

        return latent

    def run_swapper_simswap512(self, image, embedding, output):
        if not self.simswap512_model:
            self.simswap512_model = onnxruntime.InferenceSession( "./models/simswap_512_unoff.onnx", providers=self.providers)

        io_binding = self.simswap512_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='onnx::Gemm_1', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.simswap512_model.run_with_iobinding(io_binding)

    def calc_swapper_latent_ghost(self, source_embedding):
        latent = source_embedding.reshape((1,-1))

        return latent

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GhostFace-v2'):
        if swapper_model == 'GhostFace-v1':
            if not self.ghostfacev1swap_model:
                self.ghostfacev1swap_model = onnxruntime.InferenceSession( "./models/ghost_unet_1_block.onnx", providers=self.providers)

            ghostfaceswap_model = self.ghostfacev1swap_model
            output_name = '781'
            #output_name2 = 'onnx::ConvTranspose_239'
        elif swapper_model == 'GhostFace-v2':
            if not self.ghostfacev2swap_model:
                self.ghostfacev2swap_model = onnxruntime.InferenceSession( "./models/ghost_unet_2_block.onnx", providers=self.providers)

            ghostfaceswap_model = self.ghostfacev2swap_model
            output_name = '1165'
            #output_name2 = 'onnx::ConvTranspose_327'
        elif swapper_model == 'GhostFace-v3':
            if not self.ghostfacev3swap_model:
                self.ghostfacev3swap_model = onnxruntime.InferenceSession( "./models/ghost_unet_3_block.onnx", providers=self.providers)

            ghostfaceswap_model = self.ghostfacev3swap_model
            output_name = '1549'
            #output_name2 = 'onnx::ConvTranspose_415'

        io_binding = ghostfaceswap_model.io_binding()
        io_binding.bind_input(name='target', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name=output_name, device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        ghostfaceswap_model.run_with_iobinding(io_binding)

    def calc_swapper_latent_dfl(self, source_embedding):
        latent = source_embedding.reshape((1,-1))

        return latent

    def run_GFPGAN(self, image, output):
        if not self.GFPGAN_model:
            self.GFPGAN_model = onnxruntime.InferenceSession( "./models/GFPGANv1.4.onnx", providers=self.providers)

        io_binding = self.GFPGAN_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.GFPGAN_model.run_with_iobinding(io_binding)

    def run_GPEN_2048(self, image, output):
        if not self.GPEN_2048_model:
            self.GPEN_2048_model = onnxruntime.InferenceSession( "./models/GPEN-BFR-2048.onnx", providers=self.providers)

        io_binding = self.GPEN_2048_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,2048,2048), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,2048,2048), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.GPEN_2048_model.run_with_iobinding(io_binding)

    def run_GPEN_1024(self, image, output):
        if not self.GPEN_1024_model:
            self.GPEN_1024_model = onnxruntime.InferenceSession( "./models/GPEN-BFR-1024.onnx", providers=self.providers)

        io_binding = self.GPEN_1024_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,1024,1024), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,1024,1024), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.GPEN_1024_model.run_with_iobinding(io_binding)

    def run_GPEN_512(self, image, output):
        if not self.GPEN_512_model:
            self.GPEN_512_model = onnxruntime.InferenceSession( "./models/GPEN-BFR-512.onnx", providers=self.providers)

        io_binding = self.GPEN_512_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.GPEN_512_model.run_with_iobinding(io_binding)

    def run_GPEN_256(self, image, output):
        if not self.GPEN_256_model:
            self.GPEN_256_model = onnxruntime.InferenceSession( "./models/GPEN-BFR-256.onnx", providers=self.providers)

        io_binding = self.GPEN_256_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.GPEN_256_model.run_with_iobinding(io_binding)

    def run_codeformer(self, image, output, fidelity_weight_value=0.9):
        if not self.codeformer_model:
            self.codeformer_model = onnxruntime.InferenceSession( "./models/codeformer_fp16.onnx", providers=self.providers)

        io_binding = self.codeformer_model.io_binding()
        io_binding.bind_input(name='x', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        w = np.array([fidelity_weight_value], dtype=np.double)
        io_binding.bind_cpu_input('w', w)
        io_binding.bind_output(name='y', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.codeformer_model.run_with_iobinding(io_binding)

    def run_VQFR_v2(self, image, output, fidelity_ratio_value):
        if not self.VQFR_v2_model:
            self.VQFR_v2_model = onnxruntime.InferenceSession( "./models/VQFRv2.fp16.onnx", providers=self.providers)

        assert fidelity_ratio_value >= 0.0 and fidelity_ratio_value <= 1.0, 'fidelity_ratio must in range[0,1]'
        fidelity_ratio = torch.tensor(fidelity_ratio_value).to('cuda')

        io_binding = self.VQFR_v2_model.io_binding()
        io_binding.bind_input(name='x_lq', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='fidelity_ratio', device_type='cuda', device_id=0, element_type=np.float32, shape=fidelity_ratio.size(), buffer_ptr=fidelity_ratio.data_ptr())
        io_binding.bind_output('enc_feat', 'cuda')
        io_binding.bind_output('quant_logit', 'cuda')
        io_binding.bind_output('texture_dec', 'cuda')
        io_binding.bind_output(name='main_dec', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.VQFR_v2_model.run_with_iobinding(io_binding)

    def run_RestoreFormerPlusPlus(self, image, output):
        if not self.RestoreFormerPlusPlus_model:
            self.RestoreFormerPlusPlus_model = onnxruntime.InferenceSession( "./models/RestoreFormerPlusPlus.fp16.onnx", providers=self.providers)

        io_binding = self.RestoreFormerPlusPlus_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='2359', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())
        io_binding.bind_output('1228', 'cuda')
        io_binding.bind_output('1238', 'cuda')
        io_binding.bind_output('onnx::MatMul_1198', 'cuda')
        io_binding.bind_output('onnx::Shape_1184', 'cuda')
        io_binding.bind_output('onnx::ArgMin_1182', 'cuda')
        io_binding.bind_output('input.1', 'cuda')
        io_binding.bind_output('x', 'cuda')
        io_binding.bind_output('x.3', 'cuda')
        io_binding.bind_output('x.7', 'cuda')
        io_binding.bind_output('x.11', 'cuda')
        io_binding.bind_output('x.15', 'cuda')
        io_binding.bind_output('input.252', 'cuda')
        io_binding.bind_output('input.280', 'cuda')
        io_binding.bind_output('input.288', 'cuda')

        self.syncvec.cpu()
        self.RestoreFormerPlusPlus_model.run_with_iobinding(io_binding)

    def run_enhance_frame_tile_process(self, img, enhancer_type, tile_size=256, scale=1):
        _, _, height, width = img.shape

        # Calcolo del numero di tile necessari
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # Calcolo del padding necessario per adattare l'immagine alle dimensioni dei tile
        pad_right = (tile_size - (width % tile_size)) % tile_size
        pad_bottom = (tile_size - (height % tile_size)) % tile_size

        # Padding dell'immagine se necessario
        if pad_right != 0 or pad_bottom != 0:
            img = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom), 'constant', 0)

        # Creazione di un output tensor vuoto
        b, c, h, w = img.shape
        output = torch.empty((b, c, h * scale, w * scale), dtype=torch.float32, device='cuda').contiguous()

        # Selezione della funzione di upscaling in base al tipo
        upscaler_functions = {
            'RealEsrgan-x2-Plus': self.run_realesrganx2,
            'RealEsrgan-x4-Plus': self.run_realesrganx4,
            'BSRGan-x2': self.run_bsrganx2,
            'BSRGan-x4': self.run_bsrganx4,
            'UltraSharp-x4': self.run_ultrasharpx4,
            'UltraMix-x4': self.run_ultramixx4,
            'RealEsr-General-x4v3': self.run_realesrx4v3
        }

        fn_upscaler = upscaler_functions.get(enhancer_type)

        if not fn_upscaler:  # Se il tipo di enhancer non è valido
            if pad_right != 0 or pad_bottom != 0:
                img = v2.functional.crop(img, 0, 0, height, width)
            return img

        with torch.no_grad():  # Disabilita il calcolo del gradiente
            # Elaborazione dei tile
            for j in range(tiles_y):
                for i in range(tiles_x):
                    x_start, y_start = i * tile_size, j * tile_size
                    x_end, y_end = x_start + tile_size, y_start + tile_size

                    # Estrazione del tile di input
                    input_tile = img[:, :, y_start:y_end, x_start:x_end].contiguous()
                    output_tile = torch.empty((input_tile.shape[0], input_tile.shape[1], input_tile.shape[2] * scale, input_tile.shape[3] * scale), dtype=torch.float32, device='cuda').contiguous()

                    # Upscaling del tile
                    fn_upscaler(input_tile, output_tile)

                    # Inserimento del tile upscalato nel tensor di output
                    output_y_start, output_x_start = y_start * scale, x_start * scale
                    output_y_end, output_x_end = output_y_start + output_tile.shape[2], output_x_start + output_tile.shape[3]
                    output[:, :, output_y_start:output_y_end, output_x_start:output_x_end] = output_tile

            # Ritaglio dell'output per rimuovere il padding aggiunto
            if pad_right != 0 or pad_bottom != 0:
                output = v2.functional.crop(output, 0, 0, height * scale, width * scale)

        return output

    def run_realesrganx2(self, image, output):
        if not self.realesrganx2plus_model:
            self.realesrganx2plus_model = onnxruntime.InferenceSession( "./models/RealESRGAN_x2plus.fp16.onnx", providers=self.providers)

        io_binding = self.realesrganx2plus_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.realesrganx2plus_model.run_with_iobinding(io_binding)

    def run_bsrganx2(self, image, output):
        if not self.bsrganx2_model:
            self.bsrganx2_model = onnxruntime.InferenceSession( "./models/BSRGANx2.fp16.onnx", providers=self.providers)

        io_binding = self.bsrganx2_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.bsrganx2_model.run_with_iobinding(io_binding)

    def run_realesrganx4(self, image, output):
        if not self.realesrganx4plus_model:
            self.realesrganx4plus_model = onnxruntime.InferenceSession( "./models/RealESRGAN_x4plus.fp16.onnx", providers=self.providers)

        io_binding = self.realesrganx4plus_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.realesrganx4plus_model.run_with_iobinding(io_binding)

    def run_realesrx4v3(self, image, output):
        if not self.realesrx4v3_model:
            self.realesrx4v3_model = onnxruntime.InferenceSession( "./models/realesr-general-x4v3.onnx", providers=self.providers)

        io_binding = self.realesrx4v3_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.realesrx4v3_model.run_with_iobinding(io_binding)

    def run_bsrganx4(self, image, output):
        if not self.bsrganx4_model:
            self.bsrganx4_model = onnxruntime.InferenceSession( "./models/BSRGANx4.fp16.onnx", providers=self.providers)

        io_binding = self.bsrganx4_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.bsrganx4_model.run_with_iobinding(io_binding)

    def run_ultrasharpx4(self, image, output):
        if not self.ultrasharpx4_model:
            self.ultrasharpx4_model = onnxruntime.InferenceSession( "./models/4x-UltraSharp.fp16.onnx", providers=self.providers)

        io_binding = self.ultrasharpx4_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.ultrasharpx4_model.run_with_iobinding(io_binding)

    def run_ultramixx4(self, image, output):
        if not self.ultramixx4_model:
            self.ultramixx4_model = onnxruntime.InferenceSession( "./models/4x-UltraMix_Smooth.fp16.onnx", providers=self.providers)

        io_binding = self.ultramixx4_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.ultramixx4_model.run_with_iobinding(io_binding)

    def run_deoldify_artistic(self, image, output):
        if not self.deoldify_art_model:
            self.deoldify_art_model = onnxruntime.InferenceSession( "./models/ColorizeArtistic.fp16.onnx", providers=self.providers)

        io_binding = self.deoldify_art_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.deoldify_art_model.run_with_iobinding(io_binding)

    def run_deoldify_stable(self, image, output):
        if not self.deoldify_stable_model:
            self.deoldify_stable_model = onnxruntime.InferenceSession( "./models/ColorizeStable.fp16.onnx", providers=self.providers)

        io_binding = self.deoldify_stable_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.deoldify_stable_model.run_with_iobinding(io_binding)

    def run_deoldify_video(self, image, output):
        if not self.deoldify_video_model:
            self.deoldify_video_model = onnxruntime.InferenceSession( "./models/ColorizeVideo.fp16.onnx", providers=self.providers)

        io_binding = self.deoldify_video_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.deoldify_video_model.run_with_iobinding(io_binding)

    def run_ddcolor_artistic(self, image, output):
        if not self.ddcolor_art_model:
            self.ddcolor_art_model = onnxruntime.InferenceSession( "./models/ddcolor_artistic.onnx", providers=self.providers)

        io_binding = self.ddcolor_art_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.ddcolor_art_model.run_with_iobinding(io_binding)

    def run_ddcolor(self, image, output):
        if not self.ddcolor_model:
            self.ddcolor_model = onnxruntime.InferenceSession( "./models/ddcolor.onnx", providers=self.providers)

        io_binding = self.ddcolor_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.ddcolor_model.run_with_iobinding(io_binding)

    def run_occluder(self, image, output):
        if not self.occluder_model:
            self.occluder_model = onnxruntime.InferenceSession("./models/occluder.onnx", providers=self.providers)

        io_binding = self.occluder_model.io_binding()
        io_binding.bind_input(name='img', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        # torch.cuda.synchronize('cuda')
        self.syncvec.cpu()
        self.occluder_model.run_with_iobinding(io_binding)

    def run_dfl_xseg(self, image, output):
        if not self.model_xseg:
            self.model_xseg = onnxruntime.InferenceSession("./models/XSeg_model.onnx", providers=self.providers)

        io_binding = self.model_xseg.io_binding()
        io_binding.bind_input(name='in_face:0', device_type='cuda', device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='out_mask:0', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.model_xseg.run_with_iobinding(io_binding)

    def run_faceparser(self, image, output):
        if not self.faceparser_model:
            self.faceparser_model = onnxruntime.InferenceSession("./models/faceparser_fp16.onnx", providers=self.providers)

        image = image.contiguous()
        io_binding = self.faceparser_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='out', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=output.data_ptr())

        # torch.cuda.synchronize('cuda')
        self.syncvec.cpu()
        self.faceparser_model.run_with_iobinding(io_binding)

    def detect_retinaface(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles:list[int]=[0]):
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
        det_img[:new_height,:new_width,  :] = img

        # Switch to RGB and normalize
        det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()

            io_binding = self.retinaface_model.io_binding()
            io_binding.bind_input(name='input.1', device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            io_binding.bind_output('448', 'cuda')
            io_binding.bind_output('471', 'cuda')
            io_binding.bind_output('494', 'cuda')
            io_binding.bind_output('451', 'cuda')
            io_binding.bind_output('474', 'cuda')
            io_binding.bind_output('497', 'cuda')
            io_binding.bind_output('454', 'cuda')
            io_binding.bind_output('477', 'cuda')
            io_binding.bind_output('500', 'cuda')

            # Sync and run model
            syncvec = self.syncvec.cpu()
            self.retinaface_model.run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            input_height = aimg.shape[2]
            input_width = aimg.shape[3]

            fmc = 3
            center_cache = {}
            for idx, stride in enumerate([8, 16, 32]):
                scores = net_outs[idx]
                bbox_preds = net_outs[idx+fmc]
                bbox_preds = bbox_preds * stride

                kps_preds = net_outs[idx+fmc*2] * stride
                height = input_height // stride
                width = input_width // stride
                K = height * width
                key = (height, width, stride)
                if key in center_cache:
                    anchor_centers = center_cache[key]
                else:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                    anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                    if len(center_cache)<100:
                        center_cache[key] = anchor_centers

                pos_inds = np.where(scores>=score)[0]

                x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
                y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
                x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
                y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

                bboxes = np.stack([x1, y1, x2, y2], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                preds = []
                for i in range(0, kps_preds.shape[1], 2):
                    px = anchor_centers[:, i%2] + kps_preds[:, i]
                    py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                    preds.append(px)
                    preds.append(py)
                kpss = np.stack(preds, axis=-1)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                kpss.append(landmark_kpss)
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5

        return det, kpss_5, np.array(kpss)

    def detect_scrdf(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles:list[int]=[0]):
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
        det_img[:new_height,:new_width,  :] = img

        # Switch to RGB and normalize
        det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        input_name = self.scrdf_model.get_inputs()[0].name
        outputs = self.scrdf_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()

            io_binding = self.scrdf_model.io_binding()
            io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            for i in range(len(output_names)):
                io_binding.bind_output(output_names[i], 'cuda')

            # Sync and run model
            syncvec = self.syncvec.cpu()
            self.scrdf_model.run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            input_height = aimg.shape[2]
            input_width = aimg.shape[3]

            fmc = 3
            center_cache = {}
            for idx, stride in enumerate([8, 16, 32]):
                scores = net_outs[idx]
                bbox_preds = net_outs[idx+fmc]
                bbox_preds = bbox_preds * stride

                kps_preds = net_outs[idx+fmc*2] * stride
                height = input_height // stride
                width = input_width // stride
                K = height * width
                key = (height, width, stride)
                if key in center_cache:
                    anchor_centers = center_cache[key]
                else:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                    anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                    if len(center_cache)<100:
                        center_cache[key] = anchor_centers

                pos_inds = np.where(scores>=score)[0]

                x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
                y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
                x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
                y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

                bboxes = np.stack([x1, y1, x2, y2], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                preds = []
                for i in range(0, kps_preds.shape[1], 2):
                    px = anchor_centers[:, i%2] + kps_preds[:, i]
                    py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                    preds.append(px)
                    preds.append(py)
                kpss = np.stack(preds, axis=-1)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                kpss.append(landmark_kpss)
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5

        return det, kpss_5, np.array(kpss)

    def detect_yoloface(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles:list[int]=[0]):
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        # model_ratio = float(input_size[1]) / input_size[0]
        model_ratio = 1.0
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.uint8, device='cuda')
        det_img[:new_height,:new_width,  :] = img

        det_img = det_img.permute(2, 0, 1)

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = aimg.permute(1, 2, 0)
                aimg = torch.div(aimg, 255.0)
                aimg = aimg.permute(2, 0, 1)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                aimg = det_img.permute(1, 2, 0)
                aimg = torch.div(aimg, 255.0)
                aimg = aimg.permute(2, 0, 1)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
                IM = None

            io_binding = self.yoloface_model.io_binding()
            io_binding.bind_input(name='images', device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())
            io_binding.bind_output('output0', 'cuda')

            # Sync and run model
            self.syncvec.cpu()
            self.yoloface_model.run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            outputs = np.squeeze(net_outs).T

            bbox_raw, score_raw, kps_raw = np.split(outputs, [4, 5], axis=1)

            keep_indices = np.where(score_raw > score)[0]

            if keep_indices.any():
                bbox_raw, kps_raw, score_raw = bbox_raw[keep_indices], kps_raw[keep_indices], score_raw[keep_indices]

                # Compute the transformed bounding box coordinates
                x1 = bbox_raw[:, 0] - bbox_raw[:, 2] / 2
                y1 = bbox_raw[:, 1] - bbox_raw[:, 3] / 2
                x2 = bbox_raw[:, 0] + bbox_raw[:, 2] / 2
                y2 = bbox_raw[:, 1] + bbox_raw[:, 3] / 2

                # Stack the results into a single array
                bboxes_raw = np.stack((x1, y1, x2, y2), axis=-1)

                # bboxes
                if angle != 0:
                    if len(bboxes_raw) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = bboxes_raw[:, :2]  # (x1, y1)
                        points2 = bboxes_raw[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        bboxes_raw = np.hstack((points1, points2))

                kps_list = []
                for kps in kps_raw:
                    indexes = np.arange(0, len(kps), 3)
                    temp_kps = []
                    for index in indexes:
                        temp_kps.append([kps[index], kps[index + 1]])
                    kps_list.append(np.array(temp_kps))

                kpss_raw = np.stack(kps_list)

                if do_rotation:
                    for i in range(len(kpss_raw)):
                        face_size = max(bboxes_raw[i][2] - bboxes_raw[i][0], bboxes_raw[i][3] - bboxes_raw[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, kpss_raw[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            score_raw[i] = 0.0

                        if angle != 0:
                            kpss_raw[i] = faceutil.trans_points2d(kpss_raw[i], IM)

                    keep_indices = np.where(score_raw>=score)[0]
                    score_raw = score_raw[keep_indices]
                    bboxes_raw = bboxes_raw[keep_indices]
                    kpss_raw = kpss_raw[keep_indices]

                kpss_list.append(kpss_raw)
                bboxes_list.append(bboxes_raw)
                scores_list.append(score_raw)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                kpss.append(landmark_kpss)
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5

        return det, kpss_5, np.array(kpss)

    def detect_yunet(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles:list[int]=[0]):
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=False)
        img = resize(img)

        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.uint8, device='cuda')
        det_img[:new_height,:new_width,  :] = img

        # Switch to BGR
        det_img = det_img[:, :, [2,1,0]]

        det_img = det_img.permute(2, 0, 1) #3,640,640

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        input_name = self.yunet_model.get_inputs()[0].name
        outputs = self.yunet_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()
            aimg = aimg.to(dtype=torch.float32)

            io_binding = self.yunet_model.io_binding()
            io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            for i in range(len(output_names)):
                io_binding.bind_output(output_names[i], 'cuda')

            # Sync and run model
            syncvec = self.syncvec.cpu()
            self.yunet_model.run_with_iobinding(io_binding)
            net_outs = io_binding.copy_outputs_to_cpu()

            strides = [8, 16, 32]
            for idx, stride in enumerate(strides):
                cls_pred = net_outs[idx].reshape(-1, 1)
                obj_pred = net_outs[idx + len(strides)].reshape(-1, 1)
                reg_pred = net_outs[idx + len(strides) * 2].reshape(-1, 4)
                kps_pred = net_outs[idx + len(strides) * 3].reshape(
                    -1, 5 * 2)

                anchor_centers = np.stack(
                    np.mgrid[:(input_size[1] // stride), :(input_size[0] //
                                                           stride)][::-1],
                    axis=-1)
                anchor_centers = (anchor_centers * stride).astype(
                    np.float32).reshape(-1, 2)

                scores = (cls_pred * obj_pred)
                pos_inds = np.where(scores>=score)[0]

                bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[:]
                bbox_wh = np.exp(reg_pred[:, 2:]) * stride
                tl_x = (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.)
                tl_y = (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.)
                br_x = (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.)
                br_y = (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.)

                bboxes = np.stack([tl_x, tl_y, br_x, br_y], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                kpss = np.concatenate(
                    [((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + anchor_centers)
                     for i in range(5)],
                    axis=-1)

                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                kpss.append(landmark_kpss)
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5

        return det, kpss_5, np.array(kpss)

    def detect_face_landmark_5(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 512.0  / (max(w, h)*1.5)
            image, M = faceutil.transform(img, center, 512, _scale, rotate)
        else:
            image, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 512, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        image = image.permute(1,2,0)

        mean = torch.tensor([104, 117, 123], dtype=torch.float32, device='cuda')
        image = torch.sub(image, mean)

        image = image.permute(2,0,1)
        image = torch.reshape(image, (1, 3, 512, 512))

        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device='cuda')

        conf = torch.empty((1,10752,2), dtype=torch.float32, device='cuda').contiguous()
        landmarks = torch.empty((1,10752,10), dtype=torch.float32, device='cuda').contiguous()

        io_binding = self.resnet50_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='conf', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,2), buffer_ptr=conf.data_ptr())
        io_binding.bind_output(name='landmarks', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,10), buffer_ptr=landmarks.data_ptr())

        torch.cuda.synchronize('cuda')
        self.resnet50_model.run_with_iobinding(io_binding)

        scores = torch.squeeze(conf)[:, 1]
        priors = torch.tensor(self.anchors).view(-1, 4)
        priors = priors.to('cuda')

        pre = torch.squeeze(landmarks, 0)

        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)
        landmarks = torch.mul(landmarks, scale1)

        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        score=.1
        inds = torch.where(scores>score)[0]
        inds = inds.cpu().numpy()
        scores = scores.cpu().numpy()

        landmarks, scores = landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]

        if len(order) > 0:
            landmarks = landmarks[order][0]
            scores = scores[order][0]

            landmarks = np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])

            IM = faceutil.invertAffineTransform(M)
            landmarks = faceutil.trans_points2d(landmarks, IM)
            scores = np.array([scores])

            return landmarks, landmarks, scores

        return [], [], []

    def detect_face_landmark_68(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            crop_image, affine_matrix = faceutil.warp_face_by_bounding_box_for_landmark_68(img, bbox, (256, 256))
        else:
            crop_image, affine_matrix = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 256, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        crop_image = crop_image.to(dtype=torch.float32)
        crop_image = torch.div(crop_image, 255.0)
        crop_image = torch.unsqueeze(crop_image, 0).contiguous()

        io_binding = self.face_landmark_68_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32,  shape=crop_image.size(), buffer_ptr=crop_image.data_ptr())

        io_binding.bind_output('landmarks_xyscore', 'cuda')
        io_binding.bind_output('heatmaps', 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.face_landmark_68_model.run_with_iobinding(io_binding)
        net_outs = io_binding.copy_outputs_to_cpu()
        face_landmark_68 = net_outs[0]
        face_heatmap = net_outs[1]

        face_landmark_68 = face_landmark_68[:, :, :2][0] / 64.0
        face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256.0
        face_landmark_68 = cv2.transform(face_landmark_68, cv2.invertAffineTransform(affine_matrix))

        face_landmark_68 = face_landmark_68.reshape(-1, 2)
        face_landmark_68_score = np.amax(face_heatmap, axis = (2, 3))
        face_landmark_68_score = face_landmark_68_score.reshape(-1, 1)

        face_landmark_68_5, face_landmark_68_score = faceutil.convert_face_landmark_68_to_5(face_landmark_68, face_landmark_68_score)

        return face_landmark_68_5, face_landmark_68, face_landmark_68_score

    def detect_face_landmark_3d68(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 192  / (max(w, h)*1.5)
            aimg, M = faceutil.transform(img, center, 192, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=192, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = self.normalize(aimg)
        io_binding = self.face_landmark_3d68_model.io_binding()
        io_binding.bind_input(name='data', device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('fc1', 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.face_landmark_3d68_model.run_with_iobinding(io_binding)
        pred = io_binding.copy_outputs_to_cpu()[0][0]

        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if 68 < pred.shape[0]:
            pred = pred[68*-1:,:]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (192 // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (192 // 2)

        IM = faceutil.invertAffineTransform(M)
        pred = faceutil.trans_points3d(pred, IM)

        # at moment we don't use 3d points
        '''
        P = faceutil.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
        s, R, t = faceutil.P2sRt(P)
        rx, ry, rz = faceutil.matrix2angle(R)
        pose = np.array( [rx, ry, rz], dtype=np.float32 ) #pitch, yaw, roll
        '''

        # convert from 3d68 to 2d68 keypoints
        landmark2d68 = np.array(pred[:, [0, 1]])

        # convert from 68 to 5 keypoints
        landmark2d68_5, _ = faceutil.convert_face_landmark_68_to_5(landmark2d68, [])

        return landmark2d68_5, landmark2d68, []

    def detect_face_landmark_98(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            crop_image, detail = faceutil.warp_face_by_bounding_box_for_landmark_98(img, bbox, (256, 256))
        else:
            crop_image, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=256, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)
            h, w = (crop_image.size(dim=1), crop_image.size(dim=2))

        landmark = []
        landmark_5 = []
        landmark_score = []
        if crop_image is not None:
            crop_image = crop_image.to(dtype=torch.float32)
            crop_image = torch.div(crop_image, 255.0)
            crop_image = torch.unsqueeze(crop_image, 0).contiguous()

            io_binding = self.face_landmark_98_model.io_binding()
            io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32,  shape=crop_image.size(), buffer_ptr=crop_image.data_ptr())

            io_binding.bind_output('landmarks_xyscore', 'cuda')

            # Sync and run model
            syncvec = self.syncvec.cpu()
            self.face_landmark_98_model.run_with_iobinding(io_binding)
            landmarks_xyscore = io_binding.copy_outputs_to_cpu()[0]

            if len(landmarks_xyscore) > 0:
                for one_face_landmarks in landmarks_xyscore:
                    landmark_score = one_face_landmarks[:, [2]].reshape(-1)
                    landmark = one_face_landmarks[:, [0, 1]].reshape(-1,2)

                    ##recover, and grouped as [98,2]
                    if from_points == False:
                        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3] - detail[4]
                        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2] - detail[4]
                    else:
                        landmark[:, 0] = landmark[:, 0] * w
                        landmark[:, 1] = landmark[:, 1] * h

                        IM = faceutil.invertAffineTransform(M)
                        landmark = faceutil.trans_points2d(landmark, IM)

                    landmark_5, landmark_score = faceutil.convert_face_landmark_98_to_5(landmark, landmark_score)

        return landmark_5, landmark, landmark_score

    def detect_face_landmark_106(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 192  / (max(w, h)*1.5)
            #print('param:', img.size(), bbox, center, (192, 192), _scale, rotate)
            aimg, M = faceutil.transform(img, center, 192, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=192, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = self.normalize(aimg)
        io_binding = self.face_landmark_106_model.io_binding()
        io_binding.bind_input(name='data', device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('fc1', 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.face_landmark_106_model.run_with_iobinding(io_binding)
        pred = io_binding.copy_outputs_to_cpu()[0][0]

        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))

        if 106 < pred.shape[0]:
            pred = pred[106*-1:,:]

        pred[:, 0:2] += 1
        pred[:, 0:2] *= (192 // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (192 // 2)

        IM = faceutil.invertAffineTransform(M)
        pred = faceutil.trans_points(pred, IM)

        pred_5 = []
        if pred is not None:
            # convert from 106 to 5 keypoints
            pred_5 = faceutil.convert_face_landmark_106_to_5(pred)

        return pred_5, pred, []

    def detect_face_landmark_203(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 224  / (max(w, h)*1.5)

            aimg, M = faceutil.transform(img, center, 224, _scale, rotate)
        else:
            if det_kpss.shape[0] == 5:
                aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=224, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)
            else:
                aimg, M, IM = faceutil.warp_face_by_face_landmark_x(img, det_kpss, dsize=224, scale=1.5, vy_ratio=-0.1, interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = torch.div(aimg, 255.0)
        io_binding = self.face_landmark_203_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('output', 'cuda')
        io_binding.bind_output('853', 'cuda')
        io_binding.bind_output('856', 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.face_landmark_203_model.run_with_iobinding(io_binding)
        out_lst = io_binding.copy_outputs_to_cpu()
        out_pts = out_lst[2]

        out_pts = out_pts.reshape((-1, 2)) * 224.0

        if det_kpss.shape[0] == 5:
            IM = faceutil.invertAffineTransform(M)

        out_pts = faceutil.trans_points(out_pts, IM)

        out_pts_5 = []
        if out_pts is not None:
            # convert from 203 to 5 keypoints
            out_pts_5 = faceutil.convert_face_landmark_203_to_5(out_pts)

        return out_pts_5, out_pts, []

    def detect_face_landmark_478(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 256.0  / (max(w, h)*1.5)
            #print('param:', img.size(), bbox, center, (192, 192), _scale, rotate)
            aimg, M = faceutil.transform(img, center, 256, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 256, mode='arcfacemap', interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = torch.div(aimg, 255.0)
        io_binding = self.face_landmark_478_model.io_binding()
        io_binding.bind_input(name='input_12', device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('Identity', 'cuda')
        io_binding.bind_output('Identity_1', 'cuda')
        io_binding.bind_output('Identity_2', 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.face_landmark_478_model.run_with_iobinding(io_binding)
        landmarks, faceflag, blendshapes = io_binding.copy_outputs_to_cpu()
        landmarks = landmarks.reshape( (1,478,3))

        landmark = []
        landmark_5 = []
        landmark_score = []
        if len(landmarks) > 0:
            for one_face_landmarks in landmarks:
                landmark = one_face_landmarks
                IM = faceutil.invertAffineTransform(M)
                landmark = faceutil.trans_points3d(landmark, IM)
                '''
                P = faceutil.estimate_affine_matrix_3d23d(self.mean_lmk, landmark)
                s, R, t = faceutil.P2sRt(P)
                rx, ry, rz = faceutil.matrix2angle(R)
                pose = np.array( [rx, ry, rz], dtype=np.float32 ) #pitch, yaw, roll
                '''
                landmark = landmark[:, [0, 1]].reshape(-1,2)

                #get scores
                landmark_for_score = landmark[self.LandmarksSubsetIdxs]
                landmark_for_score = landmark_for_score[:, :2]
                landmark_for_score = np.expand_dims(landmark_for_score, axis=0)
                landmark_for_score = landmark_for_score.astype(np.float32)
                landmark_for_score = torch.from_numpy(landmark_for_score).to('cuda')

                io_binding_bs = self.face_blendshapes_model.io_binding()
                io_binding_bs.bind_input(name='input_points', device_type='cuda', device_id=0, element_type=np.float32,  shape=tuple(landmark_for_score.shape), buffer_ptr=landmark_for_score.data_ptr())
                io_binding_bs.bind_output('output', 'cuda')

                # Sync and run model
                syncvec = self.syncvec.cpu()
                self.face_blendshapes_model.run_with_iobinding(io_binding_bs)
                landmark_score = io_binding_bs.copy_outputs_to_cpu()[0]

                # convert from 478 to 5 keypoints
                landmark_5 = faceutil.convert_face_landmark_478_to_5(landmark)

        #return landmark, landmark_score
        return landmark_5, landmark, []

    def recognize(self, recognition_model, img, face_kps, similarity_type):
        if similarity_type == 'Optimal':
            # Find transform & Transform
            img, _ = faceutil.warp_face_by_face_landmark_5(img, face_kps, mode='arcfacemap', interpolation=v2.InterpolationMode.BILINEAR)
        elif similarity_type == 'Pearl':
            # Find transform
            dst = self.arcface_dst.copy()
            dst[:, 0] += 8.0

            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, 128, 128)
            img = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(img)
        else:
            # Find transform
            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, self.arcface_dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, 112, 112)

        if recognition_model == self.recognition_model or recognition_model == self.recognition_simswap_model:
            # Switch to BGR and normalize
            img = img.permute(1,2,0) #112,112,3
            cropped_image = img
            img = img[:, :, [2,1,0]]
            img = torch.sub(img, 127.5)
            img = torch.div(img, 127.5)
            img = img.permute(2, 0, 1) #3,112,112
        else:
            cropped_image = img.permute(1,2,0) #112,112,3
            # Converti a float32 e normalizza
            img = torch.div(img.float(), 127.5)
            img = torch.sub(img, 1)

        # Prepare data and find model parameters
        img = torch.unsqueeze(img, 0).contiguous()
        input_name = recognition_model.get_inputs()[0].name

        outputs = recognition_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        io_binding = recognition_model.io_binding()
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=img.size(), buffer_ptr=img.data_ptr())

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda')

        # Sync and run model
        self.syncvec.cpu()
        recognition_model.run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image

    def resnet50(self, image, score=.5):
        if not self.resnet50_model:
            self.resnet50_model = onnxruntime.InferenceSession("./models/res50.onnx", providers=self.providers)

            feature_maps = [[64, 64], [32, 32], [16, 16]]
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
            image_size = 512
            # re-initialize self.anchors due to clear_mem function
            self.anchors  = []

            for k, f in enumerate(feature_maps):
                min_size_array = min_sizes[k]
                for i, j in product(range(f[0]), range(f[1])):
                    for min_size in min_size_array:
                        s_kx = min_size / image_size
                        s_ky = min_size / image_size
                        dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                        dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            self.anchors += [cx, cy, s_kx, s_ky]

        image = image.permute(1,2,0)

        mean = torch.tensor([104, 117, 123], dtype=torch.float32, device='cuda')
        image = torch.sub(image, mean)

        image = image.permute(2,0,1)
        image = torch.reshape(image, (1, 3, 512, 512)).contiguous()

        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device='cuda')

        conf = torch.empty((1,10752,2), dtype=torch.float32, device='cuda').contiguous()
        landmarks = torch.empty((1,10752,10), dtype=torch.float32, device='cuda').contiguous()

        io_binding = self.resnet50_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='conf', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,2), buffer_ptr=conf.data_ptr())
        io_binding.bind_output(name='landmarks', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,10), buffer_ptr=landmarks.data_ptr())

        torch.cuda.synchronize('cuda')
        self.resnet50_model.run_with_iobinding(io_binding)

        scores = torch.squeeze(conf)[:, 1]

        priors = torch.tensor(self.anchors).view(-1, 4)
        priors = priors.to('cuda')

        pre = torch.squeeze(landmarks, 0)

        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)

        landmarks = torch.mul(landmarks, scale1)

        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        inds = torch.where(scores>score)[0]
        inds = inds.cpu().numpy()
        scores = scores.cpu().numpy()

        landmarks, scores = landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]
        landmarks = landmarks[order][0]

        return np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])

    # Face Editor
    #def get_kp_info(self, x: torch.Tensor, **kwargs) -> dict:
    def lp_motion_extractor(self, img, face_editor_type='Human-Face', **kwargs) -> dict:
        kp_info = {}
        if self.provider_name == "TensorRT-Engine":
            if face_editor_type == 'Human-Face':
                if not self.lp_motion_extractor_model:
                    if not os.path.exists("./models/liveportrait_onnx/motion_extractor.trt"):
                        onnx2trt(onnx_model_path="./models/liveportrait_onnx/motion_extractor.onnx",
                                 trt_model_path=None, precision="fp32",
                                 verbose=False
                                )
                    self.lp_motion_extractor_model = TensorRTPredictor(model_path="./models/liveportrait_onnx/motion_extractor.trt")

            motion_extractor_model = self.lp_motion_extractor_model

            # prepare_source
            I_s = torch.div(img.type(torch.float32), 255.)
            I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
            I_s = torch.unsqueeze(I_s, 0).contiguous()

            nvtx.range_push("forward")

            feed_dict = {}
            feed_dict["img"] = I_s
            preds_dict = motion_extractor_model.predict_async(feed_dict, torch.cuda.current_stream().cuda_stream)
            #preds_dict = motion_extractor_model.predict(feed_dict)
            torch.cuda.synchronize()
            kp_info = {
                'pitch': preds_dict["pitch"],
                'yaw': preds_dict["yaw"],
                'roll': preds_dict["roll"],
                't': preds_dict["t"],
                'exp': preds_dict["exp"],
                'scale': preds_dict["scale"],
                'kp': preds_dict["kp"]
            }

            nvtx.range_pop()

        else:
            if face_editor_type == 'Human-Face':
                if not self.lp_motion_extractor_model:
                    self.lp_motion_extractor_model = onnxruntime.InferenceSession("./models/liveportrait_onnx/motion_extractor.onnx", providers=self.providers)

                motion_extractor_model = self.lp_motion_extractor_model

            # prepare_source
            I_s = torch.div(img.type(torch.float32), 255.)
            I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
            I_s = torch.unsqueeze(I_s, 0).contiguous()

            pitch = torch.empty((1,66), dtype=torch.float32, device='cuda').contiguous()
            yaw = torch.empty((1,66), dtype=torch.float32, device='cuda').contiguous()
            roll = torch.empty((1,66), dtype=torch.float32, device='cuda').contiguous()
            t = torch.empty((1,3), dtype=torch.float32, device='cuda').contiguous()
            exp = torch.empty((1,63), dtype=torch.float32, device='cuda').contiguous()
            scale = torch.empty((1,1), dtype=torch.float32, device='cuda').contiguous()
            kp = torch.empty((1,63), dtype=torch.float32, device='cuda').contiguous()

            io_binding = motion_extractor_model.io_binding()
            io_binding.bind_input(name='img', device_type='cuda', device_id=0, element_type=np.float32, shape=I_s.size(), buffer_ptr=I_s.data_ptr())
            io_binding.bind_output(name='pitch', device_type='cuda', device_id=0, element_type=np.float32, shape=pitch.size(), buffer_ptr=pitch.data_ptr())
            io_binding.bind_output(name='yaw', device_type='cuda', device_id=0, element_type=np.float32, shape=yaw.size(), buffer_ptr=yaw.data_ptr())
            io_binding.bind_output(name='roll', device_type='cuda', device_id=0, element_type=np.float32, shape=roll.size(), buffer_ptr=roll.data_ptr())
            io_binding.bind_output(name='t', device_type='cuda', device_id=0, element_type=np.float32, shape=t.size(), buffer_ptr=t.data_ptr())
            io_binding.bind_output(name='exp', device_type='cuda', device_id=0, element_type=np.float32, shape=exp.size(), buffer_ptr=exp.data_ptr())
            io_binding.bind_output(name='scale', device_type='cuda', device_id=0, element_type=np.float32, shape=scale.size(), buffer_ptr=scale.data_ptr())
            io_binding.bind_output(name='kp', device_type='cuda', device_id=0, element_type=np.float32, shape=kp.size(), buffer_ptr=kp.data_ptr())

            torch.cuda.synchronize('cuda')
            motion_extractor_model.run_with_iobinding(io_binding)

            kp_info = {
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                't': t,
                'exp': exp,
                'scale': scale,
                'kp': kp
            }

        flag_refine_info: bool = kwargs.get('flag_refine_info', True)
        if flag_refine_info:
            bs = kp_info['kp'].shape[0]
            kp_info['pitch'] = faceutil.headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
            kp_info['yaw'] = faceutil.headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
            kp_info['roll'] = faceutil.headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
            kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def lp_appearance_feature_extractor(self, img, face_editor_type='Human-Face'):
        if self.provider_name == "TensorRT-Engine":
            if face_editor_type == 'Human-Face':
                if not self.lp_appearance_feature_extractor_model:
                    if not os.path.exists("./models/liveportrait_onnx/appearance_feature_extractor.trt"):
                        onnx2trt(onnx_model_path="./models/liveportrait_onnx/appearance_feature_extractor.onnx",
                                 trt_model_path=None, precision="fp16",
                                 verbose=False
                                )
                    self.lp_appearance_feature_extractor_model = TensorRTPredictor(model_path="./models/liveportrait_onnx/appearance_feature_extractor.trt")

            appearance_feature_extractor_model = self.lp_appearance_feature_extractor_model

            # prepare_source
            I_s = torch.div(img.type(torch.float32), 255.)
            I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
            I_s = torch.unsqueeze(I_s, 0).contiguous()

            nvtx.range_push("forward")

            feed_dict = {}
            feed_dict["img"] = I_s
            preds_dict = appearance_feature_extractor_model.predict_async(feed_dict, torch.cuda.current_stream().cuda_stream)
            #preds_dict = appearance_feature_extractor_model.predict(feed_dict)
            torch.cuda.synchronize()
            output = preds_dict["output"]

            nvtx.range_pop()

        else:
            if face_editor_type == 'Human-Face':
                if not self.lp_appearance_feature_extractor_model:
                    self.lp_appearance_feature_extractor_model = onnxruntime.InferenceSession("./models/liveportrait_onnx/appearance_feature_extractor.onnx", providers=self.providers)

                appearance_feature_extractor_model = self.lp_appearance_feature_extractor_model

            # prepare_source
            I_s = torch.div(img.type(torch.float32), 255.)
            I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
            I_s = torch.unsqueeze(I_s, 0).contiguous()

            output = torch.empty((1,32,16,64,64), dtype=torch.float32, device='cuda').contiguous()

            io_binding = appearance_feature_extractor_model.io_binding()
            io_binding.bind_input(name='img', device_type='cuda', device_id=0, element_type=np.float32, shape=I_s.size(), buffer_ptr=I_s.data_ptr())
            io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

            torch.cuda.synchronize('cuda')
            appearance_feature_extractor_model.run_with_iobinding(io_binding)

        return output

    def lp_retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp)
        """
        if self.provider_name == "TensorRT-Engine":
            if face_editor_type == 'Human-Face':
                if not self.lp_stitching_eye_model:
                    if not os.path.exists("./models/liveportrait_onnx/stitching_eye.trt"):
                        onnx2trt(onnx_model_path="./models/liveportrait_onnx/stitching_eye.onnx",
                                 trt_model_path=None, precision="fp16",
                                 verbose=False
                                )
                    self.lp_stitching_eye_model = TensorRTPredictor(model_path="./models/liveportrait_onnx/stitching_eye.trt")

            stitching_eye_model = self.lp_stitching_eye_model

            feat_eye = faceutil.concat_feat(kp_source, eye_close_ratio).contiguous()

            nvtx.range_push("forward")

            feed_dict = {}
            feed_dict["input"] = feat_eye
            preds_dict = stitching_eye_model.predict_async(feed_dict, torch.cuda.current_stream().cuda_stream)
            #preds_dict = stitching_eye_model.predict(feed_dict)
            torch.cuda.synchronize()
            delta = preds_dict["output"]

            nvtx.range_pop()

        else:
            if face_editor_type == 'Human-Face':
                if not self.lp_stitching_eye_model:
                    self.lp_stitching_eye_model = onnxruntime.InferenceSession("./models/liveportrait_onnx/stitching_eye.onnx", providers=self.providers)

                stitching_eye_model = self.lp_stitching_eye_model

            feat_eye = faceutil.concat_feat(kp_source, eye_close_ratio).contiguous()
            delta = torch.empty((1,63), dtype=torch.float32, device='cuda').contiguous()

            io_binding = stitching_eye_model.io_binding()
            io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=feat_eye.size(), buffer_ptr=feat_eye.data_ptr())
            io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

            torch.cuda.synchronize('cuda')
            stitching_eye_model.run_with_iobinding(io_binding)

        return delta.reshape(-1, kp_source.shape[1], 3)

    def lp_retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        Return: Bx(3*num_kp)
        """
        if self.provider_name == "TensorRT-Engine":
            if face_editor_type == 'Human-Face':
                if not self.lp_stitching_lip_model:
                    if not os.path.exists("./models/liveportrait_onnx/stitching_lip.trt"):
                        onnx2trt(onnx_model_path="./models/liveportrait_onnx/stitching_lip.onnx",
                                 trt_model_path=None, precision="fp16",
                                 verbose=False
                                )
                    self.lp_stitching_lip_model = TensorRTPredictor(model_path="./models/liveportrait_onnx/stitching_lip.trt")

            stitching_lip_model = self.lp_stitching_lip_model

            feat_lip = faceutil.concat_feat(kp_source, lip_close_ratio).contiguous()

            nvtx.range_push("forward")

            feed_dict = {}
            feed_dict["input"] = feat_lip
            preds_dict = stitching_lip_model.predict_async(feed_dict, torch.cuda.current_stream().cuda_stream)
            #preds_dict = stitching_lip_model.predict(feed_dict)
            torch.cuda.synchronize()
            delta = preds_dict["output"]

            nvtx.range_pop()

        else:
            if face_editor_type == 'Human-Face':
                if not self.lp_stitching_lip_model:
                    self.lp_stitching_lip_model = onnxruntime.InferenceSession("./models/liveportrait_onnx/stitching_lip.onnx", providers=self.providers)

                stitching_lip_model = self.lp_stitching_lip_model

            feat_lip = faceutil.concat_feat(kp_source, lip_close_ratio).contiguous()
            delta = torch.empty((1,63), dtype=torch.float32, device='cuda').contiguous()

            io_binding = stitching_lip_model.io_binding()
            io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=feat_lip.size(), buffer_ptr=feat_lip.data_ptr())
            io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

            torch.cuda.synchronize('cuda')
            stitching_lip_model.run_with_iobinding(io_binding)

        return delta.reshape(-1, kp_source.shape[1], 3)

    def lp_stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        if self.provider_name == "TensorRT-Engine":
            if face_editor_type == 'Human-Face':
                if not self.lp_stitching_model:
                    if not os.path.exists("./models/liveportrait_onnx/stitching.trt"):
                        onnx2trt(onnx_model_path="./models/liveportrait_onnx/stitching.onnx",
                                 trt_model_path=None, precision="fp16",
                                 verbose=False
                                )
                    self.lp_stitching_model = TensorRTPredictor(model_path="./models/liveportrait_onnx/stitching.trt")

            stitching_model = self.lp_stitching_model

            feat_stiching = faceutil.concat_feat(kp_source, kp_driving).contiguous()

            nvtx.range_push("forward")

            feed_dict = {}
            feed_dict["input"] = feat_stiching
            preds_dict = stitching_model.predict_async(feed_dict, torch.cuda.current_stream().cuda_stream)
            #preds_dict = stitching_model.predict(feed_dict)
            torch.cuda.synchronize()
            delta = preds_dict["output"]

            nvtx.range_pop()

        else:
            if face_editor_type == 'Human-Face':
                if not self.lp_stitching_model:
                    self.lp_stitching_model = onnxruntime.InferenceSession("./models/liveportrait_onnx/stitching.onnx", providers=self.providers)

                stitching_model = self.lp_stitching_model

            feat_stiching = faceutil.concat_feat(kp_source, kp_driving).contiguous()
            delta = torch.empty((1,65), dtype=torch.float32, device='cuda').contiguous()

            io_binding = stitching_model.io_binding()
            io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=feat_stiching.size(), buffer_ptr=feat_stiching.data_ptr())
            io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

            torch.cuda.synchronize('cuda')
            stitching_model.run_with_iobinding(io_binding)

        return delta

    def lp_stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        bs, num_kp = kp_source.shape[:2]

        kp_driving_new = kp_driving.clone()
        delta = self.lp_stitch(kp_source, kp_driving_new, face_editor_type=face_editor_type)

        delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)  # 1x20x3
        delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)  # 1x1x2

        kp_driving_new += delta_exp
        kp_driving_new[..., :2] += delta_tx_ty

        return kp_driving_new

    def lp_warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """ get the image after the warping of the implicit keypoints
        feature_3d: Bx32x16x64x64, feature volume
        kp_source: BxNx3
        kp_driving: BxNx3
        """

        if self.provider_name == "TensorRT-Engine":
            if face_editor_type == 'Human-Face':
                if not self.lp_warping_spade_fix_model:
                    if not os.path.exists("./models/liveportrait_onnx/warping_spade-fix.trt"):
                        onnx2trt(onnx_model_path="./models/liveportrait_onnx/warping_spade-fix.onnx",
                                 trt_model_path=None, precision="fp16",
                                 custom_plugin_path="./models/grid_sample_3d_plugin.dll",
                                 verbose=False
                                )
                    self.lp_warping_spade_fix_model = TensorRTPredictor(model_path="./models/liveportrait_onnx/warping_spade-fix.trt", custom_plugin_path="./models/grid_sample_3d_plugin.dll")

            warping_spade_model = self.lp_warping_spade_fix_model

            feature_3d = feature_3d.contiguous()
            kp_source = kp_source.contiguous()
            kp_driving = kp_driving.contiguous()

            nvtx.range_push("forward")

            feed_dict = {}
            feed_dict["feature_3d"] = feature_3d
            feed_dict["kp_source"] = kp_source
            feed_dict["kp_driving"] = kp_driving
            preds_dict = warping_spade_model.predict_async(feed_dict, torch.cuda.current_stream().cuda_stream)
            #preds_dict = warping_spade_model.predict(feed_dict)
            torch.cuda.synchronize()
            out = preds_dict["out"]

            nvtx.range_pop()
        else:
            if face_editor_type == 'Human-Face':
                if not self.lp_warping_spade_fix_model:
                    self.lp_warping_spade_fix_model = onnxruntime.InferenceSession("./models/liveportrait_onnx/warping_spade.onnx", providers=self.providers)

                warping_spade_model = self.lp_warping_spade_fix_model

            feature_3d = feature_3d.contiguous()
            kp_source = kp_source.contiguous()
            kp_driving = kp_driving.contiguous()

            out = torch.empty((1,3,512,512), dtype=torch.float32, device='cuda').contiguous()
            io_binding = warping_spade_model.io_binding()
            io_binding.bind_input(name='feature_3d', device_type='cuda', device_id=0, element_type=np.float32, shape=feature_3d.size(), buffer_ptr=feature_3d.data_ptr())
            io_binding.bind_input(name='kp_driving', device_type='cuda', device_id=0, element_type=np.float32, shape=kp_driving.size(), buffer_ptr=kp_driving.data_ptr())
            io_binding.bind_input(name='kp_source', device_type='cuda', device_id=0, element_type=np.float32, shape=kp_source.size(), buffer_ptr=kp_source.data_ptr())
            io_binding.bind_output(name='out', device_type='cuda', device_id=0, element_type=np.float32, shape=out.size(), buffer_ptr=out.data_ptr())

            torch.cuda.synchronize('cuda')
            warping_spade_model.run_with_iobinding(io_binding)

        return out
