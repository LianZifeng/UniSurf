import nibabel as nib
import numpy as np
from skimage.measure import label as compute_cc
from scipy.ndimage import distance_transform_edt as edt
from skimage.filters import gaussian
from tca import topology
import os

# initialize topology correction
"""
ACKNOWLEDGEMENT AND CITATION:

This fast topology correction algorithm is proposed in:
"CortexODE: Learning Cortical Surface Reconstruction by Neural ODEs".

Original Project Repository: https://github.com/m-qiang/CortexODE
Paper: Ma, Q. et al., "CortexODE: Learning Cortical Surface Reconstruction by Neural ODEs", 
       IEEE Transactions on Medical Imaging, 2022.

If you find this code useful or use it in your research, please cite the original paper:

@article{ma2022cortexode,
  title={CortexODE: Learning Cortical Surface Reconstruction by Neural ODEs},
  author={Ma, Qiang and Li, Liu and Robinson, Emma C and Kainz, Bernhard and Rueckert, Daniel and Alansary, Amir},
  journal={IEEE Transactions on Medical Imaging},
  volume={41},
  number={10},
  pages={2942--2953},
  year={2022},
  publisher={IEEE}
}
"""
topo_correct = topology()


def seg2sdf(seg, sigma=0.5):
    # ------ connected components checking ------
    cc, nc = compute_cc(seg, connectivity=2, return_num=True)
    cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i) for i in range(1, nc + 1)]))
    seg = (cc == cc_id).astype(np.float64)
    # ------ generate signed distance function ------
    sdf = -edt(seg) + edt(1 - seg)
    sdf = sdf.astype(float)
    sdf = gaussian(sdf, sigma=sigma)

    return sdf


def multiProcessing(file):
    lh_nii = nib.load(os.path.join(file, 'lh.nii.gz'))
    lh = lh_nii.get_fdata().astype(np.int32)
    affine = lh_nii.affine
    # lh[lh == 250] = 150
    lh = np.where(lh == 250, 1, 0)
    sdf = seg2sdf(lh, sigma=0.5)
    nib.save(nib.Nifti1Image(sdf, affine), os.path.join(file, 'lh.SDF.nii.gz'))


if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description='Calulate SDF from tissue map')
    parser.add_argument('--data_path', type=str, default='/your/folder/path', help='Input path')

    args = parser.parse_args()

    files = [os.path.join(args.path, f) for f in os.listdir(args.path)]
    with Pool(processes=32) as pool:

        results = list(tqdm(pool.imap_unordered(multiProcessing, files), total=len(files), desc="Processing"))
