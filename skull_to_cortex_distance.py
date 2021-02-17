"""
Calculate the distance from the edge of the cortex to the outside of the skull
at each cortical edge voxel.

Niall Duncan, 2021
"""

import os
import nibabel as ni
import numpy as np
import nipype.interfaces.fsl as fsl
import matplotlib.backends.backend_pdf as pdf
import matplotlib.gridspec as gridspec
from skimage import measure,morphology,filters
from scipy import ndimage
from scipy.spatial import distance
from joblib import Parallel, delayed


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

def calc_threshold(X,Y):
    # Fit a Gaussian and minimise distance to histogram peak
    # Based on https://www.sciencedirect.com/science/article/pii/S0010482512000157
    alpha = np.sum(Y)
    mu = 1/alpha*np.sum(X*Y)
    sigma = 1/alpha*np.sum((X-mu)**2*Y)
    G = (1/np.sqrt(2*np.pi*sigma))*np.exp(-1*((X-mu)**2/(2*sigma)))
    z = np.sum(G)
    P = (alpha/z)*G
    l = np.argmax(P-Y)
    return(X[l])

def fill_slice_holes(img_slice):
    if np.sum(img_slice) > 0: # Ignore slices with no head in it
        r = np.sin(np.exp((np.sin(img_slice)**3 + np.cos(img_slice)**2)))
        contours = measure.find_contours(r, r.max()-r.max()*.1)
        for contour in contours:
            hole = np.zeros(img_slice.shape)
            hole[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
            img_slice = img_slice+ndimage.binary_fill_holes(hole)
    img_slice[img_slice>0] = 1
    return(img_slice)

def create_head_mask(img):
    # Calculate image histogram
    intensityMax = img.max()
    histogramY,histogramX = np.histogram(img,bins=100,range=[0,intensityMax])
    histogramX = histogramX[1:]
    # Identify section with 85% of AUC
    auc_total = np.trapz(histogramY,histogramX)
    for i in range(histogramX.shape[0]):
        if np.trapz(histogramY[:i],histogramX[:i]) > auc_total*0.85:
            auc_index = i-1
            break
    # Create binary mask based on intensity threshold
    thresh = calc_threshold(histogramX[:auc_index],histogramY[:auc_index])
    headMask = img.copy()
    headMask[headMask<thresh] = 0
    headMask[headMask>0] = 1
    # Remove any small clusters remaining
    headMask = morphology.remove_small_objects(headMask.astype(bool),min_size=1000).astype(int)
    # Fill in the head
    for i in range(img.shape[2]):
        print(f'\rGenerating head mask - {((i+1)/img.shape[2]*100):0.0f}%',end='',flush=True)
        headMask[:,:,i] = fill_slice_holes(headMask[:,:,i])
        headMask[:,i,:] = fill_slice_holes(headMask[:,i,:])
    return(headMask.astype(float))

def create_brain_mask(img,head_mask,img_dims):
    print(f'\nGenerating brain mask')
    # Calculate image histogram within head
    intensityMax = img[head_mask==1].max()
    histogramY,histogramX = np.histogram(img[head_mask==1],bins=100,range=[0,intensityMax])
    histogramX = histogramX[1:]
    # Identify sections with lower 80% of AUC
    auc_total = np.trapz(histogramY,histogramX)
    for i in range(histogramX.shape[0]):
        if np.trapz(histogramY[:i],histogramX[:i]) > auc_total*0.8:
            lower_index = i-1
            break
    # Calculate upper and lower thresholds
    thresh_l = calc_threshold(histogramX[:lower_index],histogramY[:lower_index])
    thresh_u = calc_threshold(histogramX[lower_index:],histogramY[lower_index:])
    # Do initial masking
    brain_mask = img.copy()
    brain_mask[brain_mask<thresh_l] = 0
    brain_mask[brain_mask>thresh_u] = 0
    brain_mask[brain_mask>0] = 1
    # Erode image with 2mm spherical structuring element
    sphere_2mm = morphology.ball(int(np.round(2/min(img_dims),0)))
    tmp = ndimage.binary_erosion(brain_mask,structure=sphere_2mm)
    # Detect largest contiguous component
    labels = measure.label(tmp)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    # Dilate with 3mm spherical structuring element and close gaps
    sphere_3mm = morphology.ball(int(np.round(3/min(img_dims),0)))
    brain_mask = ndimage.binary_dilation(largestCC,structure=sphere_3mm)
    brain_mask = ndimage.binary_closing(brain_mask,structure=sphere_3mm)
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    return(brain_mask.astype(int))

def get_slice_edge(img_slice):
    edge_mask = filters.sobel(img_slice)
    edge_mask[edge_mask>0] = 1
    return(edge_mask)

def create_mask_edge(mask,img_dims):
    edge_mask = np.zeros(mask.shape)
    temp = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        if np.sum(mask[i,:,:]) > 0:
            temp[i,:,:] = get_slice_edge(mask[i,:,:])
    edge_mask = edge_mask + temp
    temp = np.zeros(mask.shape)
    for i in range(mask.shape[1]):
        if np.sum(mask[:,i,:]) > 0:
            temp[:,i,:] = get_slice_edge(mask[:,i,:])
    edge_mask = edge_mask + temp
    edge_mask[edge_mask>0] = 1
    edge_mask[mask==0] = 0
    return(edge_mask)

def calc_min_distance(coords,targets):
    return(distance.cdist(np.array([coords]),targets,metric='euclidean').min())

def calc_distances(brain_edge,head_edge):
    brain_coords = np.transpose(np.nonzero(brain_edge))
    head_coords = np.transpose(np.nonzero(head_edge))
    temp = Parallel(n_jobs=-2,verbose=0)(delayed(calc_min_distance)(brain_coords[i,:],
            head_coords)for i in range(brain_coords.shape[0]))
    dist_img = np.zeros(brain_edge.shape)
    for i in range(brain_coords.shape[0]):
        dist_img[brain_coords[i,0],brain_coords[i,1],brain_coords[i,2]] = temp[i]
    return(dist_img)

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

def find_nearest(mat,coord):
    if mat[coord[0],coord[1],coord[2]] == 0:
        x,y,z = np.nonzero(mat)
        min_idx = ((x-coord[0])**2 + (y-coord[1])**2 + (z-coord[2])**2).argmin()
        return((x[min_idx],y[min_idx],z[min_idx]))
    else:
        return((coord[0],coord[1],coord[2]))


def extract_distance(brain_dist,coord,radius):
    # Make spherical mask
    dims = brain_dist.shape
    X, Y, Z = np.ogrid[:dims[0], :dims[1], :dims[2]]
    dist_from_center = np.sqrt((X - coord[0])**2 + (Y-coord[1])**2 + (Z-coord[2])**2)
    sphere = dist_from_center <= radius
    mask = sphere.astype(int)
    # Get average distance within mask
    mask[brain_dist==0] = 0
    avg_dist = np.mean(brain_dist[mask==1])
    sd_dist = np.std(brain_dist[mask==1])
    return((avg_dist,sd_dist),sphere)

def plot_results(t1_img,brain_edge,head_edge,fname,spheres=None,dist_result=None,coords_dist=None):
    # Ensure masks are integers
    brain_edge = brain_edge.astype(int)
    head_edge = head_edge.astype(int)
    # Slices to show
    x_slices = np.round(np.array((t1_img.shape[0]/5,2*t1_img.shape[0]/5,t1_img.shape[0]/2,
                3*t1_img.shape[0]/5,4*t1_img.shape[0]/5)),0).astype(int)
    out_pdf = pdf.PdfPages(fname)
    fig = plt.figure(constrained_layout=True,figsize=(11,8))
    gs = fig.add_gridspec(3,5)
    for i,slice in enumerate(x_slices):
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(ndimage.rotate(t1_img[slice,:,:],90),cmap='gray')
        ax.imshow(np.ma.masked_where(ndimage.rotate(head_edge[slice,:,:],90)==0,
                    ndimage.rotate(head_edge[slice,::],90)),cmap='viridis')
        ax.imshow(np.ma.masked_where(ndimage.rotate(brain_edge[slice,:,:],90)==0,
                    ndimage.rotate(brain_edge[slice,::],90)),cmap='viridis_r')
        ax.tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                        left=False,labelleft=False)
    if not spheres is None:
        if isinstance(spheres,list):
            cnt = 0
            for i,sphere in enumerate(spheres):
                ax = fig.add_subplot(gs[1,cnt])
                ax.imshow(ndimage.rotate(t1_img[coords_dist[i][0],:,:],90),cmap='gray')
                ax.imshow(np.ma.masked_where(ndimage.rotate(sphere[coords_dist[i][0],:,:],90)==0,
                        ndimage.rotate(sphere[coords_dist[i][0],:,:],90)),cmap='viridis_r')
                ax.tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                                left=False,labelleft=False)
                ax = fig.add_subplot(gs[1,cnt+1])
                ax.imshow(ndimage.rotate(t1_img[:,coords_dist[i][1],:],90),cmap='gray')
                ax.imshow(np.ma.masked_where(ndimage.rotate(sphere[:,coords_dist[i][1],:],90)==0,
                        ndimage.rotate(sphere[:,coords_dist[i][1],:],90)),cmap='viridis_r')
                ax.tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                                left=False,labelleft=False)
                ax = fig.add_subplot(gs[2,cnt:cnt+2])
                ax.text(0.1,0.9,f'Region {i+1}',fontsize=14)
                plusminus = "\u00B1"
                ax.text(0.1,0.8,f"Distance = {dist_result[i][0]:0.1f} {plusminus} {dist_result[i][1]:0.1f} mm",fontsize=14)
                ax.axis('off')
                cnt = cnt+2
        else:
            ax = fig.add_subplot(gs[1,0])
            ax.imshow(ndimage.rotate(t1_img[coords_dist[0],:,:],90),cmap='gray')
            ax.imshow(np.ma.masked_where(ndimage.rotate(spheres[coords_dist[0],:,:],90)==0,
                        ndimage.rotate(spheres[coords_dist[0],::],90)),cmap='viridis_r')
            ax.tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                        left=False,labelleft=False)
            ax = fig.add_subplot(gs[1,1])
            ax.imshow(ndimage.rotate(t1_img[:,coords_dist[1],:],90),cmap='gray')
            ax.imshow(np.ma.masked_where(ndimage.rotate(spheres[:,coords_dist[1],:],90)==0,
                        ndimage.rotate(spheres[:,coords_dist[1],:],90)),cmap='viridis_r')
            ax.tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                        left=False,labelleft=False)
            ax = fig.add_subplot(gs[2,0:2])
            ax.text(0.1,0.9,'Region 1',fontsize=14)
            plusminus = "\u00B1"
            ax.text(0.1,0.8,f"Distance = {dist_result[0]:0.1f} {plusminus} {dist_result[1]:0.1f} mm",fontsize=14)
            ax.axis('off')
    out_pdf.savefig()
    out_pdf.close()
    plt.close()

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
    t1_img,t1_aff,t1_dims = load_nifti(in_file)
    isotropic = True

    # If images are anisotropic resample to largest dimension size
    if len(np.unique(t1_dims)) > 1:
        print('Resampling image')
        isotropic = False
        t1_img_orig = t1_img.copy()
        t1_dims_orig = t1_dims.copy()
        t1_img,t1_dims,resample_aff = resample_img(t1_img,t1_dims)

    # Calculate head mask + edge
    head_mask = create_head_mask(t1_img)
    head_edge = create_mask_edge(head_mask,t1_dims)

    # Extract brain tissue and detect edge
    brain_mask = create_brain_mask(t1_img,head_mask,t1_dims)
    brain_edge = create_mask_edge(brain_mask,t1_dims)

    # Calculate distance between brain and scalp - convert from voxels to mm
    brain_dist = calc_distances(brain_edge,head_edge)
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
    save_nifti(head_edge,t1_aff,os.path.join(out_dir,'head_edge.nii.gz'))
    save_nifti(brain_edge,t1_aff,os.path.join(out_dir,'brain_edge.nii.gz'))
    save_nifti(brain_dist,t1_aff,os.path.join(out_dir,'scalp_distance.nii.gz'))

    # If coordinates are provided calculate distances at them
    if not coords is None:
        # Test if an MNI to anat matrix was provided. Do alignment to standard
        # space if no affine provided.
        # Get the input voxels in anatomical space.
        if trans_file:
            anat_coords = convert_coords(coords,trans_file,in_file,t1_aff)
        else:
            run_mni_alignment(in_file,out_dir)
            trans_file = os.path.join(out_dir,'mni_to_anat.mat')
            anat_coords = convert_coords(coords,trans_file,in_file,t1_aff)
        # Identify the nearest voxels in the distance image
        # Get distance from sphere around these voxels
        radius = np.round(10/t1_dims.min(),0)
        if anat_coords.ndim == 1:
            coords_dist = find_nearest(brain_dist,anat_coords)
            dist_result,spheres = extract_distance(brain_dist,coords_dist,radius)
        else:
            coords_dist = [ [] for _ in range(anat_coords.shape[0])]
            dist_result = [ [] for _ in range(anat_coords.shape[0])]
            spheres = [ [] for _ in range(anat_coords.shape[0])]
            for i,coord in enumerate(anat_coords):
                coords_dist[i] = find_nearest(brain_dist,coord)
                dist_result[i],spheres[i] = extract_distance(brain_dist,coords_dist[i],radius)
        # Generate output
        print('Plotting images')
        fname = os.path.join(out_dir,'skull_to_cortex_distance_results.pdf')
        plot_results(t1_img,brain_edge,head_edge,fname,spheres=spheres,
                        dist_result=dist_result,coords_dist=coords_dist)
    else:
        # Generate output
        print('Plotting images')
        plot_results(t1_img,brain_edge,head_edge,fname)


scalp_distance('sub-03',coords=coords)
