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
    slices.sort(key=lambda s: int(s.InstanceNumber))
    vol = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)

    # Apply slope/intercept
    for i, s in enumerate(slices):
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", 0)
        vol[i] = vol[i]*slope + intercept

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
    im = im.resize((size,size), Image.BILINEAR)
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

    slice_idx = st.slider("Slice index", 0, vol.shape[0]-1, vol.shape[0]//2)
    img = normalize_slice(vol[slice_idx])

    if nrrd_file:
        seg = load_nrrd(nrrd_file)
        # Resize segmentation to match slice
        seg_slice = np.array(Image.fromarray(seg[slice_idx]).resize((img.shape[1], img.shape[0]), Image.NEAREST))
        img = overlay_segmentation(img, seg_slice)

    st.image(resize_img(img), caption=f"Slice {slice_idx}", use_column_width=True)
