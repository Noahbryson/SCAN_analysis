import numpy as np

import mne
from mne.datasets import fetch_fsaverage

misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"
fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # downloads if needed
raw = mne.io.read_raw(misc_path / "seeg" / "sample_seeg_ieeg.fif")
epochs = mne.Epochs(raw, detrend=1, baseline=None)
epochs = epochs["Response"][0]  # just process one epoch of data for speed
montage = epochs.get_montage()
this_subject_dir = misc_path / "seeg"
head_mri_t = mne.coreg.estimate_head_mri_t("sample_seeg", this_subject_dir)
montage.apply_trans(head_mri_t)
mri_mni_t = mne.read_talxfm("sample_seeg", misc_path / "seeg")
montage.apply_trans(mri_mni_t)  # mri to mni_tal (MNI Taliarach)
montage.apply_trans(mne.transforms.Transform(fro="mni_tal", to="mri", trans=np.eye(4)))
epochs.set_montage(montage)
trans = mne.channels.compute_native_head_t(montage)
view_kwargs = dict(azimuth=105, elevation=100, focalpoint=(0, 0, -15))
# brain = mne.viz.Brain(
#     "fsaverage",
#     subjects_dir=subjects_dir,
#     cortex="low_contrast",
#     alpha=0.25,
#     background="white",
# )
# brain.add_sensors(epochs.info, trans=trans)
# brain.add_head(alpha=0.25, color="tan")
# brain.show_view(distance=400, **view_kwargs)
mne.viz.set_3d_backend('pyvistaqt')
# brain = mne.viz.Brain(
#     "fsaverage", subjects_dir=subjects_dir, surf="flat", background="black"
# )
# brain.add_annotation("aparc.a2009s")
# brain.add_sensors(epochs.info, trans=trans)



aseg = "aparc+aseg"  # parcellation/anatomical segmentation atlas
labels, colors = mne.get_montage_volume_labels(
    montage, "fsaverage", subjects_dir=subjects_dir, aseg=aseg
)

# separate by electrodes which have names like LAMY 1
electrodes = set(
    [
        "".join([lttr for lttr in ch_name if not lttr.isdigit() and lttr != " "])
        for ch_name in montage.ch_names
    ]
)
print(f"Electrodes in the dataset: {electrodes}")

# electrodes = ("LPM", "LSMA")  # choose two for this example
electrodes = ('RPHP', 'LENT', 'LPHG', 'RAHP', 'LPLI', 'LTPO', 'LAMY', 'LPIT', 'LOFC', 'LBRI', 'LSMA', 'LPM', 'LACN', 'LPCN', 'LSTG')
for elec in electrodes:
    picks = [ch_name for ch_name in epochs.ch_names if elec in ch_name]
    fig, ax = mne.viz.plot_channel_labels_circle(labels, colors, picks=picks)
    fig.text(0.3, 0.9, "Anatomical Labels", color="white")