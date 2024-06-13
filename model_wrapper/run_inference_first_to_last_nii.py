import argparse
import os, sys, fnmatch
import numpy as np
import SimpleITK as sitk

from apply_model import inference_test


def find(pattern, path):
    """
    Find filenames in a given directory matching specified pattern
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def write_file(mask,dir_name,file_name,input_img):
    """ Write mask to NIfTI file """
    if not os.path.exists(dir_name):
    	os.makedirs(dir_name)	

    out_file = os.path.join(dir_name,file_name + '.nii.gz')
    mask = np.flip(mask, 2)
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(input_img)
    sitk.WriteImage(mask_img, out_file)


def main(args):

  print(args)
  if num_args == 1: #singularitycontainer
    inputScanPath = '/scratch/inputNii/'
    inputMaskPath = os.path.join(inputScanPath, 'Masks')
    outputPath = '/scratch/outputNii/'
    modelDir = '/software/model/'
  else:
    inputScanPath = args.inputPath
    inputMaskPath = os.path.join(inputScanPath, 'Masks')
    outputPath = args.outputPath
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    wrapperDir = os.path.abspath(os.path.join(scriptDir, os.pardir))
    modelDir = os.path.abspath(os.path.join(wrapperDir, "models"))

  reg_model_path = os.path.join(modelDir, 'sv_reg_model_reg.pt')
  seg_model_path = os.path.join(modelDir, 'sv_seg_model_seg.pt')
  print('---- Model location ---')
  print(reg_model_path)
  print(seg_model_path)

  # Identify inputs
  keyword = 'SCAN'
  src_img_path = find('*_MR SCAN_first_scan_3D.nii.gz', inputScanPath)
  src_mask_path = find('*_MR SCAN_first_4D.nii.gz', inputMaskPath)
  target_img_path = find('*_MR SCAN_last_scan_3D.nii.gz', inputScanPath)

  if not src_img_path:
    raise ValueError('Missing src img ''*_MR SCAN_first_scan_3D.nii.gz''')
  else:
    print('INFO: src_img_path : ' + src_img_path[0])
  if not src_mask_path:
    raise ValueError('Missing mask ''*_MR SCAN_first_4D.nii.gz''')
  else:
    print('INFO: src_mask_path : ' + src_mask_path[0])
  if not target_img_path:
    raise ValueError('Missing target img ''*_MR SCAN_last_scan_3D.nii.gz''')
  else:
    print('INFO: target_img_path : ' + target_img_path[0])

  # Apply model
  target_img, result_seg, result_dvf = inference_test(src_img_path[0],\
                                       src_mask_path[0], target_img_path[0],\
                                       reg_model_path, seg_model_path, args)


  # Save mask to file
  print("Writing mask to file...")
  input_fname = src_img_path[0]
  pt_name = os.path.basename(input_fname)
  pt_mask_name = pt_name.replace(keyword, 'MASK')

  mask_filename = os.path.join(outputPath, pt_mask_name)
  print(mask_filename)
  write_file(result_seg, outputPath, pt_mask_name, target_img)

  # Save DVF to file
  flowX = result_dvf[:,:,:,1] #* 10 #Cols
  flowY = result_dvf[:,:,:,0] #* 10 #Rows
  flowZ = result_dvf[:,:,:,2] #* 10 #Slices
  flow = np.stack((flowX,flowY,flowZ),axis=3)

  dvf_path = os.path.join(outputPath, 'DVF')
  dvf_filename = pt_name.replace(keyword, 'DVF')
  if not os.path.isdir(dvf_path):
  # Create directory to write DVF output
      os.mkdir(dvf_path)

  dvf_filepath = os.path.join(dvf_path, dvf_filename)
  print(dvf_filepath)
  write_file(flow, dvf_path, dvf_filename, target_img)

if __name__=="__main__":

     # Parse input arguments
    parser = argparse.ArgumentParser()
    num_args = len(sys.argv)

    # data organization parameters
    parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')

    # network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')

    # loss hyperparameters
    parser.add_argument('--image-loss', default='mse',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                        help='weight of deformation loss (default: 0.01)')
    parser.add_argument('--smooth', type=float, default=40, help='weight of deformation loss (default: 0.01)')
    parser.add_argument('--flownum', type=int, default=7, help='flow number (default: 8)')

    # for output
    parser.add_argument('--svdir', type=str, default='tep_output_deformation',
                        help='weight of deformation loss (default: 0.01)')

    # I/O
    parser.add_argument('--inputPath', type=str, help='Path to input NIfTI files')
    parser.add_argument('--outputPath', type=str, help='Path to output NIfTI files')
    opt, unknown = parser.parse_known_args()

    if num_args == 3: # Conda archive
       opt.inputPath = sys.argv[1]
       opt.outputPath = sys.argv[2]
         
    main(opt)
