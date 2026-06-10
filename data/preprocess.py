import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from pathlib import Path
from tqdm import tqdm

from data.label_maps import harmonize_labels


def preprocess_volume(nifti_path, target_spacing=(1.0, 1.0, 1.0), is_mask=False):
    img          = sitk.ReadImage(str(nifti_path))
    orig_spacing = img.GetSpacing()
    orig_size    = img.GetSize()

    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)

    return sitk.GetArrayFromImage(resample.Execute(img))  # (D, H, W)


def normalize(volume):
    mask = volume > 0
    out  = np.zeros_like(volume, dtype=np.float32)
    if mask.any():
        out[mask] = (volume[mask] - volume[mask].mean()) / (volume[mask].std() + 1e-8)
    return out


def crop_or_pad(volume, target=(128, 128, 128)):
    for dim in range(3):
        src, tgt = volume.shape[dim], target[dim]
        if src >= tgt:
            start  = (src - tgt) // 2
            volume = np.take(volume, range(start, start + tgt), axis=dim)
        else:
            pb = (tgt - src) // 2
            pa = tgt - src - pb
            pw = [(0, 0)] * 3
            pw[dim] = (pb, pa)
            volume = np.pad(volume, pw)
    return volume


def get_boundary_mask(seg_mask, radius=1):
    struct     = np.ones((2 * radius + 1,) * 3, dtype=bool)
    tumor_mask = seg_mask > 0
    eroded     = binary_erosion(tumor_mask, structure=struct)
    return (tumor_mask & ~eroded).astype(np.uint8)


def process_patient(patient_dir, dataset_name, out_dir,
                    modalities=None, target=(128, 128, 128)):
    if modalities is None:
        modalities = ["t1", "t1ce", "t2", "flair"]

    patient_id     = Path(patient_dir).name
    timepoint_dirs = sorted(Path(patient_dir).iterdir())
    days_list      = list(range(len(timepoint_dirs)))  # fallback — replace with real dates if available

    for t_idx, tp_dir in enumerate(timepoint_dirs):
        vols = []
        for mod in modalities:
            nii_path = next(tp_dir.glob(f"*{mod}*.nii*"), None)
            if nii_path is None:
                print(f"  Missing {mod} for {patient_id} t={t_idx}, skipping timepoint")
                break
            vol = preprocess_volume(nii_path, is_mask=False)
            vol = normalize(vol)
            vol = crop_or_pad(vol, target)
            vols.append(vol)
        else:
            image_4ch = np.stack(vols, axis=0).astype(np.float32)  # (4, D, H, W)

            mask_path = next(tp_dir.glob("*seg*.nii*"), None)
            if mask_path:
                mask     = preprocess_volume(mask_path, is_mask=True)
                mask     = harmonize_labels(mask.astype(np.int16), dataset_name)
                mask     = crop_or_pad(mask, target)
                boundary = get_boundary_mask(mask)
            else:
                mask     = np.zeros(target, dtype=np.int16)
                boundary = np.zeros(target, dtype=np.uint8)

            save_path = Path(out_dir) / dataset_name / patient_id / f"t{t_idx:02d}.npz"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                save_path,
                image=image_4ch,
                mask=mask,
                boundary=boundary,
                days=np.array([days_list[t_idx]])
            )

    print(f"Done: {patient_id} — {len(timepoint_dirs)} timepoints")


def run_preprocessing(raw_dir, dataset_name, out_dir):
    patient_dirs = sorted(Path(raw_dir).iterdir())
    print(f"\nProcessing {dataset_name}: {len(patient_dirs)} patients")
    for p_dir in tqdm(patient_dirs):
        if p_dir.is_dir():
            process_patient(p_dir, dataset_name, out_dir)
