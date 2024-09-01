
from dfl.DFMModel import DFMModel 
import rope.FaceUtil as faceutil
from skimage import transform as trans
from torchvision.transforms import v2
import torch
import numpy as np
from torchvision import transforms
from math import floor, ceil


def calculate_transform(self, kps, parameters):
    if parameters['FaceSwapperModelTextSel'] != 'GhostFace-v1' and parameters['FaceSwapperModelTextSel'] != 'GhostFace-v2' and parameters['FaceSwapperModelTextSel'] != 'GhostFace-v3':
        dst = self.arcface_dst * 4.0
        dst[:,0] += 32.0

        if parameters['FaceAdjSwitch']:
            dst[:,0] += parameters['KPSXSlider']
            dst[:,1] += parameters['KPSYSlider']
            dst[:,0] -= 255
            dst[:,0] *= (1+parameters['KPSScaleSlider']/100)
            dst[:,0] += 255
            dst[:,1] -= 255
            dst[:,1] *= (1+parameters['KPSScaleSlider']/100)
            dst[:,1] += 255

        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst)
    else:
        dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')

        if parameters['FaceAdjSwitch']:
            for k in dst:
                k[:,0] += parameters['KPSXSlider']
                k[:,1] += parameters['KPSYSlider']
                k[:,0] -= 255
                k[:,0] *= (1+parameters['KPSScaleSlider']/100)
                k[:,0] += 255
                k[:,1] -= 255
                k[:,1] *= (1+parameters['KPSScaleSlider']/100)
                k[:,1] += 255

        M, _ = faceutil.estimate_norm_arcface_template(kps, src=dst)
        tform = trans.SimilarityTransform()
        tform.params[0:2] = M

    return tform
            
def get_scaling_transforms():
    # Scaling Transforms
    t512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
    t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
    t128 = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)                
    return t512, t256, t128

def get_resized_faces(img, tform, t512, t256, t128):
    original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=v2.InterpolationMode.BILINEAR )
    original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)# 3, 512, 512
    original_face_256 = t256(original_face_512)
    original_face_128 = t128(original_face_256)
    return original_face_512, original_face_256, original_face_128

def get_input_face_latent_and_dim(self, s_e, t_e, parameters, swapper_model, dfl_model, original_face_512, original_face_256, original_face_128):
    if dfl_model:
        latent = []
        # latent = torch.from_numpy(self.models.calc_swapper_latent_dfl(s_e)).float().to('cuda')
        input_face_affined = original_face_512
        dim = 4

    else:
        if swapper_model == 'Inswapper128':
            latent = torch.from_numpy(self.models.calc_swapper_latent(s_e)).float().to('cuda')
            if parameters['FaceLikenessSwitch']:
                factor = parameters['FaceLikenessFactorSlider']
                dst_latent = torch.from_numpy(self.models.calc_swapper_latent(t_e)).float().to('cuda')
                latent = latent - (factor * dst_latent)

            dim = 1
            if parameters['SwapperTypeTextSel'] == '128':
                dim = 1
                input_face_affined = original_face_128
            elif parameters['SwapperTypeTextSel'] == '256':
                dim = 2
                input_face_affined = original_face_256
            elif parameters['SwapperTypeTextSel'] == '512':
                dim = 4
                input_face_affined = original_face_512
        elif swapper_model == 'SimSwap512':
            latent = torch.from_numpy(self.models.calc_swapper_latent_simswap512(s_e)).float().to('cuda')
            if parameters['FaceLikenessSwitch']:
                factor = parameters['FaceLikenessFactorSlider']
                dst_latent = torch.from_numpy(self.models.calc_swapper_latent_simswap512(t_e)).float().to('cuda')
                latent = latent - (factor * dst_latent)

            dim = 4
            input_face_affined = original_face_512
        elif swapper_model == 'GhostFace-v1' or swapper_model == 'GhostFace-v2' or swapper_model == 'GhostFace-v3':
            latent = torch.from_numpy(self.models.calc_swapper_latent_ghost(s_e)).float().to('cuda')
            if parameters['FaceLikenessSwitch']:
                factor = parameters['FaceLikenessFactorSlider']
                dst_latent = torch.from_numpy(self.models.calc_swapper_latent_ghost(t_e)).float().to('cuda')
                latent = latent - (factor * dst_latent)
            dim = 2
            input_face_affined = original_face_256
    return input_face_affined, latent, dim


def process_in_swapper(self, input_face_affined, latent, swapper_model, itex, dim, output, prev_face):
    for _ in range(itex):
        for j in range(dim // 128):
            for i in range(dim // 128):
                input_face_disc = input_face_affined[j::dim // 128, i::dim // 128].permute(2, 0, 1).unsqueeze(0).contiguous()

                swapper_output = torch.empty((1, 3, 128, 128), dtype=torch.float32, device='cuda').contiguous()
                self.models.run_swapper(input_face_disc, latent, swapper_output)

                swapper_output = swapper_output.squeeze().permute(1, 2, 0)

                output[j::dim // 128, i::dim // 128] = swapper_output.clone()

        prev_face = input_face_affined.clone()
        input_face_affined = output.clone()
        output = torch.mul(output, 255).clamp(0, 255)

    return input_face_affined, output, prev_face, dim

def process_sim_swap(self, input_face_affined, latent, swapper_model, itex, dim, output, prev_face):
    for _ in range(itex):
        input_face_disc = input_face_affined.permute(2, 0, 1).unsqueeze(0).contiguous()
        swapper_output = torch.empty((1, 3, 512, 512), dtype=torch.float32, device='cuda').contiguous()
        self.models.run_swapper_simswap512(input_face_disc, latent, swapper_output)

        swapper_output = swapper_output.squeeze().permute(1, 2, 0)

        prev_face = input_face_affined.clone()
        input_face_affined = swapper_output.clone()

        output = swapper_output.clone()
        output = torch.mul(output, 255).clamp(0, 255)

    return input_face_affined, output, prev_face, dim

def process_ghost_face(self, input_face_affined, latent, swapper_model, itex, dim, output, prev_face):
    for _ in range(itex):
        input_face_disc = torch.mul(input_face_affined, 255.0).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).contiguous()

        swapper_output = torch.empty((1, 3, 256, 256), dtype=torch.float32, device='cuda').contiguous()
        self.models.run_swapper_ghostface(input_face_disc, latent, swapper_output, swapper_model)

        swapper_output = swapper_output[0].permute(1, 2, 0)
        swapper_output = torch.mul(swapper_output, 127.5).add(127.5)

        prev_face = input_face_affined.clone()
        input_face_affined = swapper_output.clone().div(255)

        output = swapper_output.clone().clamp(0, 255)

    return input_face_affined, output, prev_face, dim

def process_face_swapping(self, input_face_affined, latent, swapper_model, itex):
    dim_map = {
        'Inswapper128': 128,
        'SimSwap512': 512,
        'GhostFace-v1': 256,
        'GhostFace-v2': 256,
        'GhostFace-v3': 256,
    }
    
    dim = dim_map.get(swapper_model, 128)
    prev_face = None
    output = torch.zeros_like(input_face_affined)

    # Select the appropriate processing function based on the swapper model
    if swapper_model == 'Inswapper128':
        input_face_affined, output, prev_face, dim = process_in_swapper(self, input_face_affined, latent, swapper_model, itex, dim, output, prev_face)
    elif swapper_model == 'SimSwap512':
        input_face_affined, output, prev_face, dim = process_sim_swap(self, input_face_affined, latent, swapper_model, itex, dim, output, prev_face)
    elif swapper_model.startswith('GhostFace'):
        input_face_affined, output, prev_face, dim = process_ghost_face(self, input_face_affined, latent, swapper_model, itex, dim, output, prev_face)

    return input_face_affined, output, prev_face, dim

def process_dfl_swap(dfl_model, input_face_affined, original_face_512, parameters ):
    # Get face alignment image processor
    fai_ip = dfl_model.get_fai_ip(original_face_512.permute(1, 2, 0).cpu().numpy())
    swap_image = fai_ip.get_image('HWC')
    # Convert and obtain outputs
    out_celeb, out_celeb_mask, out_face_mask = dfl_model.convert(swap_image, parameters['DFLAmpMorphSlider']/100, rct=parameters['DFLRCTColorSwitch'])

    swapper_output = torch.from_numpy(out_celeb.copy()).cuda()
    # swapper_output = swapper_output.permute(1, 2, 0)
    prev_face = input_face_affined.clone()
    input_face_affined = swapper_output.clone()

    output = swapper_output.clone()
    return input_face_affined, output, prev_face

def process_swap_strength(swap, prev_face, original_face_512, itex, t512, parameters):
    if itex == 0:
        swap = original_face_512.clone()
    else:
        alpha = np.mod(parameters['StrengthSlider'], 100)*0.01
        if alpha==0:
            alpha=1
        # Blend the images
        prev_face = torch.mul(prev_face, 255)
        prev_face = torch.clamp(prev_face, 0, 255)
        prev_face = prev_face.permute(2, 0, 1)
        prev_face = t512(prev_face)
        swap = torch.mul(swap, alpha)
        prev_face = torch.mul(prev_face, 1-alpha)
        swap = torch.add(swap, prev_face)
    
    return swap, prev_face

def process_border_mask(parameters, device):

    # Create border mask
    border_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
    border_mask = torch.unsqueeze(border_mask,0)
    # if parameters['BorderState']:
    top = parameters['BorderTopSlider']
    left = parameters['BorderLeftSlider']
    right = 128-parameters['BorderRightSlider']
    bottom = 128-parameters['BorderBottomSlider']

    border_mask[:, :top, :] = 0
    border_mask[:, bottom:, :] = 0
    border_mask[:, :, :left] = 0
    border_mask[:, :, right:] = 0

    gauss = transforms.GaussianBlur(parameters['BorderBlurSlider']*2+1, (parameters['BorderBlurSlider']+1)*0.2)
    border_mask = gauss(border_mask)

    return border_mask, gauss

def process_swap_color(swap, parameters, device):
    swap = torch.unsqueeze(swap,0).contiguous()
    swap = v2.functional.adjust_gamma(swap, parameters['ColorGammaSlider'], 1.0)
    swap = torch.squeeze(swap)
    swap = swap.permute(1, 2, 0).type(torch.float32)
    del_color = torch.tensor([parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']], device=device)
    swap += del_color
    swap = torch.clamp(swap, min=0., max=255.)
    swap = swap.permute(2, 0, 1).type(torch.uint8)
    swap = v2.functional.adjust_brightness(swap, parameters['ColorBrightSlider'])
    swap = v2.functional.adjust_contrast(swap, parameters['ColorContrastSlider'])
    swap = v2.functional.adjust_saturation(swap, parameters['ColorSaturationSlider'])
    swap = v2.functional.adjust_sharpness(swap, parameters['ColorSharpnessSlider'])
    swap = v2.functional.adjust_hue(swap, parameters['ColorHueSlider'])
    return swap

def process_swap_merge(img, swap, swap_mask, tform,):
    # Cslculate the area to be mergerd back to the original frame
    IM512 = tform.inverse.params[0:2, :]
    corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])

    x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
    y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])

    left = floor(np.min(x))
    if left<0:
        left=0
    top = floor(np.min(y))
    if top<0:
        top=0
    right = ceil(np.max(x))
    if right>img.shape[2]:
        right=img.shape[2]
    bottom = ceil(np.max(y))
    if bottom>img.shape[1]:
        bottom=img.shape[1]

    # Untransform the swap
    swap = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
    swap = v2.functional.affine(swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0,interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
    swap = swap[0:3, top:bottom, left:right]
    swap = swap.permute(1, 2, 0)
    #cv2.imwrite('test_swap512_3.png', cv2.cvtColor(swap.cpu().numpy(), cv2.COLOR_RGB2BGR))

    # Untransform the swap mask
    swap_mask = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
    swap_mask = v2.functional.affine(swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
    swap_mask = swap_mask[0:1, top:bottom, left:right]
    swap_mask = swap_mask.permute(1, 2, 0)
    swap_mask = torch.sub(1, swap_mask)

    # Apply the mask to the original image areas
    img_crop = img[0:3, top:bottom, left:right]
    img_crop = img_crop.permute(1,2,0)
    img_crop = torch.mul(swap_mask,img_crop)

    #Add the cropped areas and place them back into the original image
    swap = torch.add(swap, img_crop)
    swap = swap.type(torch.uint8)
    swap = swap.permute(2,0,1)
    return swap, top, bottom, left, right