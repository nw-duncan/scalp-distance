

import nipype.interfaces.fsl as fsl
from skimage import measure,morphology,filters


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
