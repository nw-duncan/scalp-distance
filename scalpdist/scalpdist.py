"""
Calculate the distance from the edge of the cortex to the outside of the skull
at each cortical edge voxel.

Niall Duncan, 2021
"""

import os
import numpy as np
from scipy import ndimage
from scalpdist import anat,plotting,algo

def scalp_distance(subject,trans_file=None,coords=None):
    """
    Calculate the distance from the grey matter surface to the scalp from a T1
    image. Will calculate for specific coordinate(s) if provided.

    Assumes data conforms to BIDs format. Output to ./derivatives/scalp_distance
    folder.

    Parameters
    ------------

    subject:        Subject ID
    trans_file:     File containing affine transformation from MNI to native space.
                    Assumes the final column is the offset. If None, this will
                    be calculated
    coords:         Coordinates in MNI space (mm) where stimulation is applied.
                    If None, no thickness will be output

    """

    print(f'Processing scalp distance for {subject}')

    # Check if output directory exists
    root_dir = os.getcwd()
    out_dir = os.path.join(root_dir,'derivatives','scalp_distance',subject)
    if not os.path.isdir(out_dir):
        if not os.path.isdir(os.path.join(root_dir,'derivatives')):
            os.mkdir(os.path.join(root_dir,'derivatives'))
        if not os.path.isdir(os.path.join(root_dir,'derivatives','scalp_distance')):
            os.mkdir(os.path.join(root_dir,'derivatives','scalp_distance'))
        if not os.path.isdir(os.path.join(root_dir,'derivatives','scalp_distance',subject)):
            os.mkdir(os.path.join(root_dir,'derivatives','scalp_distance',subject))

    # Read in T1 image
    in_file = os.path.join(root_dir,'rawdata',subject,'anat',subject+'_T1w.nii.gz')
    if not os.path.isfile(in_file):
        print('Anatomical image not found')
        print('Exiting')
        #return
    t1_img,t1_aff,t1_dims = anat.load_nifti(in_file)
    isotropic = True

    # If images are anisotropic resample to largest dimension size
    if len(np.unique(t1_dims)) > 1:
        print('Resampling image')
        isotropic = False
        t1_img_orig = t1_img.copy()
        t1_dims_orig = t1_dims.copy()
        t1_img,t1_dims,resample_aff = anat.resample_img(t1_img,t1_dims)

    # Calculate head mask + edge
    head_mask = algo.create_head_mask(t1_img)
    head_edge = algo.create_mask_edge(head_mask,t1_dims)

    # Extract brain tissue and detect edge
    brain_mask = algo.create_brain_mask(t1_img,head_mask,t1_dims)
    brain_edge = algo.create_mask_edge(brain_mask,t1_dims)

    # Calculate distance between brain and scalp - convert from voxels to mm
    brain_dist = algo.calc_distances(brain_edge,head_edge)
    brain_dist *= t1_dims[0]

    # If necessary, return resampled image to original space
    if not isotropic:
        brain_dist = ndimage.affine_transform(brain_dist,matrix=np.linalg.inv(resample_aff[:3,:3]),
            output_shape=t1_img_orig.shape,order=0)
        head_edge = ndimage.affine_transform(head_edge,matrix=np.linalg.inv(resample_aff[:3,:3]),
            output_shape=t1_img_orig.shape,order=0)
        brain_edge = ndimage.affine_transform(brain_edge,matrix=np.linalg.inv(resample_aff[:3,:3]),
            output_shape=t1_img_orig.shape,order=0)
        t1_img = t1_img_orig.copy()
        t1_dims = t1_dims_orig.copy()
        t1_img = t1_img_orig.copy()

    # Save brain images
    print('\nGenerating NIFTI images')
    anat.save_nifti(head_edge,t1_aff,os.path.join(out_dir,'head_edge.nii.gz'))
    anat.save_nifti(brain_edge,t1_aff,os.path.join(out_dir,'brain_edge.nii.gz'))
    anat.save_nifti(brain_dist,t1_aff,os.path.join(out_dir,'scalp_distance.nii.gz'))

    # If coordinates are provided calculate distances at them
    if not coords is None:
        # Test if an MNI to anat matrix was provided. Do alignment to standard
        # space if no affine provided.
        # Get the input voxels in anatomical space.
        if trans_file:
            anat_coords = algo.convert_coords(coords,trans_file,in_file,t1_aff)
        else:
            anat.run_mni_alignment(in_file,out_dir)
            trans_file = os.path.join(out_dir,'mni_to_anat.mat')
            anat_coords = algo.convert_coords(coords,trans_file,in_file,t1_aff)
        # Identify the nearest voxels in the distance image
        # Get distance from sphere around these voxels
        radius = np.round(10/t1_dims.min(),0)
        if anat_coords.ndim == 1:
            coords_dist = algo.find_nearest(brain_dist,anat_coords)
            dist_result,spheres = algo.extract_distance(brain_dist,coords_dist,radius)
        else:
            coords_dist = [ [] for _ in range(anat_coords.shape[0])]
            dist_result = [ [] for _ in range(anat_coords.shape[0])]
            spheres = [ [] for _ in range(anat_coords.shape[0])]
            for i,coord in enumerate(anat_coords):
                coords_dist[i] = algo.find_nearest(brain_dist,coord)
                dist_result[i],spheres[i] = algo.extract_distance(brain_dist,coords_dist[i],radius)
        # Generate output
        print('Plotting images')
        fname = os.path.join(out_dir,'skull_to_cortex_distance_results.pdf')
        plotting.plot_results(t1_img,brain_edge,head_edge,fname,spheres=spheres,
                        dist_result=dist_result,coords_dist=coords_dist)
    else:
        # Generate output
        print('Plotting images')
        plotting.plot_results(t1_img,brain_edge,head_edge,fname)
