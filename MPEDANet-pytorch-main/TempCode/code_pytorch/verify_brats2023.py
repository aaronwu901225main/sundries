import os
import pathlib
import SimpleITK as sitk

root = "../dataset/brats2023_part_2/data"

bad = []
all_cases = sorted([str(p) for p in pathlib.Path(root).glob('BraTS-GLI-*')])

for case_dir in all_cases:
    base = os.path.basename(case_dir)
    files = [
        os.path.join(case_dir, f"{base}-seg.nii"),
        os.path.join(case_dir, f"{base}-t1c.nii"),
        os.path.join(case_dir, f"{base}-t1n.nii"),
        os.path.join(case_dir, f"{base}-t2f.nii"),
        os.path.join(case_dir, f"{base}-t2w.nii"),
    ]
    ok = True
    for fp in files:
        try:
            if (not os.path.exists(fp)) or os.path.getsize(fp) == 0:
                ok = False
                break
            r = sitk.ImageFileReader()
            r.SetFileName(fp)
            r.ReadImageInformation()
        except Exception:
            ok = False
            break
    if not ok:
        bad.append(base)

print(f"Scanned {len(all_cases)} cases. Bad cases: {len(bad)}")
for b in bad:
    print(b)
