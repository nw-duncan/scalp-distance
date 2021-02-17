"""
Functions for working with anatomical images.

Includes reading and writing nifti files, interfacing with FSL, and modifying
anatomical images where necessary.

"""

import nibabel as ni
import nipype.interfaces.fsl as fsl

def load_nifti(fname):
    img = ni.load(fname)
    aff = img.affine
    vox_dims = np.array(img.header.get_zooms())
    return(img.get_fdata(),aff,vox_dims)

def save_nifti(img,aff,fname):
    ni.Nifti1Image(img,aff).to_filename(fname)

def resample_img(t1_img,t1_dims):
    max_loc = np.argmax(t1_dims)
    scale = np.zeros((3,3))
    output_shape = np.zeros(3)
    for i in range(3):
        if i == max_loc:
            scale[i,i] = 1
            output_shape[i] = t1_img.shape[i]
        else:
            scale[i,i] = t1_dims[i]*t1_dims.max()
            output_shape[i] = np.round(t1_img.shape[i]/t1_dims.max(),0)
    offset = np.zeros((1,3))
    target_affine = np.hstack((scale,offset.T))
    temp = ndimage.affine_transform(t1_img,target_affine,output_shape=output_shape.astype(int))
    return(temp,np.repeat(t1_dims.max(),3),target_affine)

def run_mni_alignment(in_file,out_dir):
    print('Aligning anatomical with MNI template')
    flirt = fsl.FLIRT()
    flirt.inputs.in_file = in_file
    flirt.inputs.reference = os.path.join(os.environ['FSLDIR'],'data','standard',
            'MNI152_T1_2mm.nii.gz')
    flirt.inputs.out_file = os.path.join(out_dir,'T1w_mni.nii.gz')
    flirt.inputs.out_matrix_file = os.path.join(out_dir,'anat_to_mni.mat')
    flirt.run()
    # Invert transform
    invt = fsl.ConvertXFM()
    invt.inputs.in_file = os.path.join(out_dir,'anat_to_mni.mat')
    invt.inputs.invert_xfm = True
    invt.inputs.out_file = os.path.join(out_dir,'mni_to_anat.mat')
    invt.run()

def convert_coords(coords,trans_file,dest_file,t1_aff):
    # Send coordinates to a text file
    txt_file = os.path.join(out_dir,'input_mni_coordinates_mm.txt')
    np.savetxt(txt_file,coords)
    # Convert coordinates to T1 space
    out_file = os.path.join(out_dir,'input_anat_coordinates_mm.txt')
    warp = fsl.WarpPoints()
    warp.inputs.in_coords = txt_file
    warp.inputs.src_file = os.path.join(os.environ['FSLDIR'],'data','standard',
            'MNI152_T1_2mm.nii.gz')
    warp.inputs.dest_file = dest_file
    warp.inputs.xfm_file = trans_file
    warp.inputs.coord_mm = True
    warp.inputs.out_file = out_file
    warp.run()
    # Convert from mm to voxel coordinates
    mm = np.loadtxt(out_file)
    if mm.ndim > 1:
        vox = []
        for temp in mm:
            vox.append(np.dot(np.linalg.inv(t1_aff),np.hstack((temp,1))))
        vox = np.round(np.array(vox),0)[:,:3].astype(int)
    else:
        vox = np.dot(np.linalg.inv(t1_aff),np.hstack((mm,1)))
        vox = np.round(np.array(vox),0)[:3].astype(int)
    np.savetxt(os.path.join(out_dir,'input_anat_coordinates_vox.txt'),vox)
    return(vox)
