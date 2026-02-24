import os
import tempfile
import zipfile
import numpy as np
import streamlit as st
import pydicom
import nrrd
from PIL import Image

# ----------------------
# Helper functions
# ----------------------

def load_dicom_zip(zip_file):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(tmpdir)

    # Recursively find all .dcm files
    files = []
    for root, _, filenames in os.walk(tmpdir):
        for f in filenames:
            if f.lower().endswith(".dcm"):
                files.append(os.path.join(root, f))

    if not files:
        raise ValueError("No DICOM files found in zip")

    slices = [pydicom.dcmread(f) for f in files]

    # Sort slices by ImagePositionPatient[2] or SliceLocation
    def get_slice_position(ds):
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "SliceLocation"):
            return float(ds.SliceLocation)
        else:
            return 0
    slices.sort(key=get_slice_position)

    # Use the first slice as reference shape
    target_shape = (slices[0].Rows, slices[0].Columns)
    vol_slices = []
    
    for s in slices:
        img = s.pixel_array.astype(np.float32)
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", 0)
        img = img * slope + intercept
    
        # Normalize to 0-255 for PIL
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    
        # Resize ALL slices to target_shape
        img_resized = Image.fromarray(img_norm.astype(np.uint8))
        img_resized = img_resized.resize(target_shape[::-1], Image.BILINEAR)
        img_resized = np.array(img_resized).astype(np.float32)
    
        vol_slices.append(img_resized)
    
    vol = np.stack(vol_slices, axis=0)
    return vol

def load_nrrd(file):
    data, _ = nrrd.read(file)
    return data.astype(int)

def normalize_slice(slc):
    slc = np.nan_to_num(slc)
    return (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)

def overlay_segmentation(img, mask, alpha=0.4):
    base = (img*255).astype(np.uint8)
    rgb = np.stack([base]*3, axis=-1).astype(np.float32)
    rgb[mask>0] = (1-alpha)*rgb[mask>0] + alpha*np.array([255,0,0])
    return np.clip(rgb/255.0, 0, 1)

def resize_img(img, size=256):
    im = Image.fromarray((img*255).astype(np.uint8))
    im = im.resize((size, size), Image.BILINEAR)
    return np.array(im)/255.0

# ----------------------
# Streamlit UI
# ----------------------

st.title("DICOM Zip + NRRD Viewer")

dicom_zip = st.file_uploader("Upload DICOM.zip", type="zip")
nrrd_file = st.file_uploader("Upload Segmentation (.nrrd)", type="nrrd")

if dicom_zip:
    vol = load_dicom_zip(dicom_zip)
    st.write("Volume shape:", vol.shape)

    # Slider to pick slice
    slice_idx = st.slider("Slice index", 0, vol.shape[0]-1, vol.shape[0]//2)
    img = normalize_slice(vol[slice_idx])

    if nrrd_file:
        seg = load_nrrd(nrrd_file)
        # Resize segmentation to match slice dimensions
        seg_slice = np.array(
            Image.fromarray(seg[slice_idx]).resize((img.shape[1], img.shape[0]), Image.NEAREST)
        )
        img = overlay_segmentation(img, seg_slice)

    st.image(resize_img(img), caption=f"Slice {slice_idx}", use_column_width=True)
