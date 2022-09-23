# Installation
The package can be installed via pip as follows:

```
python -m pip install https://github.com/nw-duncan/scalp-distance/archive/main.tar.gz
```

# Data organisation
The scripts assume that your data are in folders organised in a manner similar to the BIDS standard (https://bids.neuroimaging.io/).  

Output files will be created in a new folder within the "derivatives" directory.

# Usage
All functionality is accessed through the `scalp_distance` function.

Import this via the scalpdist package.

```
from scalpdist import scalpdist
```

It is assumed that processing is being run from the root directory within which the rawdata is located.

### Calculate distance map only
When all that is required is the map of distances around the edge of the brain, the function requires only the subject ID:

```
scalpdist.scalp_distance('sub-01')
```

This will create the relevant NIFTI files in the derivatives folder.

### Calculate distances at a specific MNI coordinate
The mean distance between scalp and brain around a coordinate or set of coordinates at which you intend to stimulate can be calculated. This distance is the average in a 1cm radius sphere around the coordinate.

Coordinates should be provided in MNI152 mm format. 

#### Without a pre-calulated linear transform
The alignment between MNI and subject space will be calculated if no transformation is provided.

```
scalpdist.scalp_distance('sub-01',coords=[68,-18,32])
```

More than one set of coordinates can be entered as follows:

```
scalpdist.scalp_distance('sub-01',coords=([68,-18,32],[30,-100,6]))
```

#### With a pre-calculated linear transform
An existing linear transformation between MNI and subject space can be used. This should be compatible with FSL. The path to this file must be entered:

```
scalpdist.scalp_distance('sub-01',trans_file='derivatives/preprocess/sub-01/anat/mni_to_anat.mat',coords=[68,-18,32])
```

# Dependencies

## Required python packages
- Numpy
- Matplotlib
- Scipy
- Nibabel
- Scikit-Image
- Joblib
- Nipype

## Other required tools
- FSL
