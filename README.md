# MRI_Pancreas_Fullshot_AnatomicCtxShape
This repository provides pre-trained deep learning models [1] for longitudinal registration and segmentation of organs at risk (OARs)
in radiotherapy treatment planning of pancreatic cancer patients.  
  
It operates on axial T2w-MRI scans acquired for this purpose in the head-first supine (HFS) orientation. This model requires the scans to be rigidly registered, and input segmentations of the OARs listed below to be available on the baseline (earlier) scan. Pre-processing including automatic extraction of patient outline and resizing [2] are performed using pyCERR.
  
## Outputs

### OARs

* Liver  
* Bowel_Lg  
* Bowel_Sm   
* Duostomach  

### Deformable vector field
DVF for deforming the earlier (moving) scan to a later (baseline) scan.
  
## Installing dependencies  
Dependencies specified in `requirements.txt` may be installed as follows:  
  
````
conda create -y --name MRI_Pancreas_Fullshot_AnatomicCtxShape python=3.8
conda activate MRI_Pancreas_Fullshot_AnatomicCtxShape  
pip install -r requirements.txt  
````
  
## Applying the model  
```  
python run_inference_first_to_last_nii.py <input_nii_directory> <output_nii_directory>    
```
  
A Jupyter [notebook](https://github.com/cerr/pyCERR-Notebooks/blob/main/auto_register_segment_MR_Pancreas_OARs.ipynb) demonstrating how to run the model and visualize auto-segmented structures is provided.
      
## Citing this work  
You may publish material involving results produced using this software provided that you reference the following  
   
1. Jiang, J., Hong, J., Tringale, K., Reyngold, M., Crane, C., Tyagi, N., & Veeraraghavan, H. (2023). Progressively refined deep joint registration segmentation (ProRSeg) of gastrointestinal organs at risk: Application to MRI and cone-beam CT. *Medical Physics*, 50(8), 4758-4774.  
  
2.  Iyer, A., Locastro, E., Apte, A. P., Veeraraghavan, H., & Deasy, J. O. (2021). Portable framework to deploy deep learning segmentation models for medical images. *bioRxiv*, 2021-03.    

## License
By downloading the software you are agreeing to the following terms and conditions as well as to the Terms of Use of CERR software.

    THE SOFTWARE IS PROVIDED "AS IS" AND CERR DEVELOPMENT TEAM AND ITS COLLABORATORS DO NOT MAKE ANY WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.
        
    This software is for research purposes only and has not been approved for clinical use.
    
    Software has not been reviewed or approved by the Food and Drug Administration, and is for non-clinical, IRB-approved Research Use Only. In no event shall data or images generated through the use of the Software be used in the provision of patient care.
    
    You may publish papers and books using results produced using software provided that you reference the appropriate citations (https://doi.org/10.1016/j.phro.2020.05.009, https://doi.org/10.1118/1.1568978, https://doi.org/10.1002/mp.13046, https://doi.org/10.1101/773929)
    
    YOU MAY NOT DISTRIBUTE COPIES of this software, or copies of software derived from this software, to others outside your organization without specific prior written permission from the CERR development team except where noted for specific software products.

    All Technology and technical data delivered under this Agreement are subject to US export control laws and may be subject to export or import regulations in other countries. You agree to comply strictly with all such laws and regulations and acknowledge that you have the responsibility to obtain such licenses to export, re-export, or import as may be required after delivery to you.


