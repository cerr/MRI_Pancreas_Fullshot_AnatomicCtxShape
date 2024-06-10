#!/usr/bin/env python
import os
import SimpleITK as sitk
import numpy as np
import torch
from batchgenerators.augmentations.utils import pad_nd_image

os.environ['VXM_BACKEND'] = 'pytorch' # Import voxelmorph with pytorch backend
import voxelmorph as vxm

# Input dimensions
inshape = (128, 192, 128)

# Output labels
num_labels = 4

# Preprocessing parameters (histogram matching)
num_levels = 128  # uint32(128)
num_match_points = 32  # unit32(32)

# Initialize loss variables
plot_loss_value = []
plot_loss_value1 = []
plot_loss_value2 = []

def load_scan(file_name):
    """ Read NIfTI image """
    input_img = sitk.ReadImage(file_name)
    scan = sitk.GetArrayFromImage(input_img)
    scan = np.moveaxis(scan,0,-1)
    return scan, input_img

def load_mask(file_name):
    """ Read NIfTI mask """
    input_mask = sitk.ReadImage(file_name)
    mask = sitk.GetArrayFromImage(input_mask)
    mask = np.moveaxis(mask, 0, -2)

    mask_dims = np.shape(mask) #This is a 4-D mask
    label_map = np.zeros(mask_dims[:-1])
    for label in range(mask_dims[-1]):
        str_mask = np.squeeze(mask[:,:,:,label]).astype(bool)
        label_map[str_mask] = label + 1
    return label_map

def label_to_bin(label_map):
    """Convert label map to binary mask stack"""
    label_siz = np.shape(label_map)
    out_siz = label_siz + (num_labels,)
    bin_mask = np.zeros(out_siz)

    for label in range(num_labels):
        bin_mask[:,:,:,label] = label_map==(label+ 1)

    return bin_mask


def process_input_data(mri1_img, mri1_msk, mri2_img):
    """Image standardization"""
    mri1_img[mri1_img < 0] = 0
    mri1_img[mri1_img > 2000] = 2000

    mri1_img = mri1_img * 1. / 2000

    mri2_img[mri2_img < 0] = 0
    mri2_img[mri2_img > 2000] = 2000
    mri2_img = mri2_img * 1. / 2000

    Planimg = mri1_img
    CBCTimg = mri2_img

    Planimg_msk = mri1_msk

    Planimg_msk[Planimg_msk > 4] = 0

    Planimg = np.squeeze(Planimg)
    CBCTimg = np.squeeze(CBCTimg)
    Planimg_msk = np.squeeze(Planimg_msk)

    Planimg = pad_nd_image(Planimg, (128, 192, 128))
    CBCTimg = pad_nd_image(CBCTimg, (128, 192, 128))
    Planimg_msk = pad_nd_image(Planimg_msk, (128, 192, 128))

    Planimg = Planimg[None, :, :]
    CBCTimg = CBCTimg[None, :, :]

    Planimg_msk = Planimg_msk[None, :, :]

    return Planimg, Planimg_msk, CBCTimg  # ,CBCTimg_msk


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def inference_test(src_img_path, src_mask_path, target_img_path, \
                   reg_model_path, seg_model_path, args):
    """Apply trained model"""

    # Load input images
    source_img, __ = load_scan(src_img_path)
    source_msk = load_mask(src_mask_path)
    target_img, __ = load_scan(target_img_path)
    print('INFO: raw source size', source_img.shape)
    orig_size = source_img.shape

    # Pre-processing
    ### Histogram equalization
    ref_img_path = os.path.join(os.path.dirname(__file__), 'Abdo_CP_pre_fx1img_img_128.nii')

    ref_img = sitk.ReadImage(ref_img_path, imageIO="NiftiImageIO")
    ref_img_array = sitk.GetArrayFromImage(ref_img)
    ref_img_float64 = sitk.GetImageFromArray(ref_img_array.astype(float))

    source_img_itk = sitk.GetImageFromArray(source_img)
    target_img_itk = sitk.GetImageFromArray(target_img)
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(num_levels)
    matcher.SetNumberOfMatchPoints(num_match_points)
    matcher.ThresholdAtMeanIntensityOn()
    matched_img_src = matcher.Execute(source_img_itk, ref_img_float64)
    src_img_equalized = sitk.GetArrayFromImage(matched_img_src)
    matched_img_target = matcher.Execute(target_img_itk, ref_img_float64)
    target_img_equalized = sitk.GetArrayFromImage(matched_img_target)

    ### Standardization
    source_img, source_msk, target_img = process_input_data(src_img_equalized,\
                                         source_msk, target_img_equalized)
    print('INFO: after processing source size', source_img.shape)

    source_img = torch.from_numpy(source_img)
    source_msk = torch.from_numpy(source_msk)
    target_img = torch.from_numpy(target_img)

    source_img = source_img.unsqueeze(0)
    source_msk = source_msk.unsqueeze(0)
    target_img = target_img.unsqueeze(0)
    print('INFO: ', source_img.shape)

    smooth_w = args.smooth
    bidir = args.bidir
    bidir = False

    # Enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    # Unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    # Load model
    model = vxm.networks.VxmDense_3D_LSTM(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )
    if torch.cuda.device_count() and torch.cuda.is_available():
        print('Using GPU')
        print('GPU device count: ', torch.cuda.device_count())
        device = torch.device("cuda:0")
        model = model.load(reg_model_path, device)
        model.to(device)
    else:
        print('Using CPU')
        device = torch.device('cpu')

    model_seg3d = vxm.networks.UNet3D_Seg_LSTM(in_channels=1 + 1 + 4, out_channels=4 + 1, final_sigmoid=False)
    model_seg3d.load_state_dict(torch.load(seg_model_path))
    model_seg3d = model_seg3d.cuda()

    flow_ini = torch.zeros(1, 3, 128, 192, 128).cuda()
    grid_template_ini = torch.zeros(1, 1, 128, 192, 128)
    grid_w = 12
    for i in range(0, 32):
        grid_template_ini[:, :, :, (i + 0) * 6, :] = 1

    for i in range(0, 32):
        grid_template_ini[:, :, (i + 0) * 4, :, :] = 1

    grid_template_ini = grid_template_ini.cuda()

    #grid = generate_grid(inshape)
    #grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()
    #grid = grid.permute(0, 4, 2, 3, 1)

    # Apply model
    flow_num = args.flownum
    # No gradient calculation
    with torch.no_grad():  # no gradient calculation

        cbct_val_img = target_img.float().cuda()
        plan_ct_img = source_img.float().cuda()
        planct_val_msk = source_msk.float().cuda()

        cbct_val_img_show = torch.squeeze(cbct_val_img)
        cbct_val_img_show = cbct_val_img_show.data.cpu().numpy()

        plan_ct_img_show = torch.squeeze(plan_ct_img)
        plan_ct_img_show = plan_ct_img_show.data.cpu().numpy()

        plan_ct_img_show = torch.squeeze(planct_val_msk)
        plan_ct_img_show = plan_ct_img_show.data.cpu().numpy()

        # Feed the data in
        for seg_iter_val in range(0, flow_num + 1):
            if seg_iter_val == 0:
                states = None

                y_pred_val, _, _, _, _, states, y_m_pred_val,\
                    grid_template, flow_abs, flow_cur, flow = \
                    model.forward_seg_training_with_grid_deformation(\
                    plan_ct_img, cbct_val_img, planct_val_msk, 0,\
                    states, grid_template_ini, flow_ini,\
                    plan_ct_img, planct_val_msk)
            else:
                y_pred_val, _, _, _, _, states, y_m_pred_val, grid_template, flow_abs, flow_cur, flow = model.forward_seg_training_with_grid_deformation(
                    y_pred_val, cbct_val_img, y_m_pred_val, 0, states, grid_template, flow, plan_ct_img,
                    planct_val_msk)

            if seg_iter_val == 0:
                state_seg = None

        'Multi_channel PlanCT'
        y_m_pred_val_mt = torch.zeros((y_m_pred_val.size(0), 4, 128, 192, 128))

        for organ_index in range(1, 5):
            temp_target = torch.zeros(y_m_pred_val.size())
            temp_target[y_m_pred_val == organ_index] = 1

            y_m_pred_val_mt[:, organ_index - 1, :, :, :] = torch.squeeze(temp_target)

        y_m_pred_val_mt = y_m_pred_val_mt.cuda()
        seg_in_val = torch.cat((cbct_val_img, y_pred_val), 1)
        seg_in_val = torch.cat((seg_in_val, y_m_pred_val_mt), 1)

        seg_result, h_seg, c_seg = model_seg3d(seg_in_val, state_seg)
        seg_result = torch.argmax(seg_result, dim=1)

        seg_result_show = torch.squeeze(seg_result)
        seg_result_show = seg_result_show.data.cpu().numpy()
        seg_result_show = seg_result_show[:, :, 0:orig_size[2]]
        seg_result_show = np.int8(seg_result_show)

        state_seg=[h_seg,c_seg]

    # Convert 3D label map to 4D stack of binary masks
    seg_result_bin = label_to_bin(seg_result_show)
    target_img = sitk.GetImageFromArray(np.squeeze(target_img.data.cpu().numpy()))

    # Flow: Permute so that 1st axis is row, 2nd axis is col and 3rd is slice deformation
    flow_array = flow.permute(0, 2, 3, 4, 1)
    flow_array = flow_array.cpu().numpy()
    flow_array = np.squeeze(flow_array)

    # Visualize results
    # import matplotlib.pyplot as plt
    # curr_fx = target_img.data.cpu().numpy()
    # prev_fx = source_img.data.cpu().numpy()
    # def_img = y_pred_val.data.cpu().numpy()
    # fig, ax = plt.subplots(3)
    # slc = 40
    # ax[0].imshow(curr_fx[0,0,:, :, slc], cmap="gray", alpha=1)
    # ax[0].imshow(seg_result_show[:,:, slc], cmap="jet", alpha=0.5)
    # ax[1].imshow(prev_fx[0,0,:, :, slc], cmap="gray", alpha=1)
    # ax[2].imshow(def_img[0,0,:, :, slc], cmap="gray", alpha=1)
    # ax[2].imshow(seg_result_show[:,:, slc], cmap="jet", alpha=0.5)
    # diffM = curr_fx[0,0,:, :, slc] - def_img[0,0,:,:, slc]
    # plt.imshow(diffM,cmap="jet")
    # plt.colorbar()
    # plt.imshow(def_img[0,0,:, :, slc], cmap="gray", alpha=1)
    # plt.imshow(seg_result_show[:,:, slc], cmap="jet", alpha=0.5)

    return target_img, seg_result_bin, flow_array
