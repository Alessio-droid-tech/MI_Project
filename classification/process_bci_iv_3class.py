import os
import numpy as np
import scipy.io as sio
import mne
from config_bci_iv import SFREQ, EEG_CHANNELS, EPOCH_TMAX, EPOCH_TMIN

# Directory separate dai file binari originali
DATA_DIR   = os.path.join("..", "dataset_bci_iv")
OUTPUT_DIR = os.path.join("..", "data", "clean_bci_iv_3class")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_epochs_3class(filepath):
    """
    Estrae epoche per Left Hand (1), Right Hand (2) e Feet (3).
    La classe Tongue (4) viene scartata.
    Mapping etichette: Left=0, Right=1, Feet=2
    """
    print(f"Caricamento {os.path.basename(filepath)}...")

    mat     = sio.loadmat(filepath)
    mi_runs = mat['data'][0, 3:9]

    epochs_data_list = []
    events_list      = []
    current_event_id = 0

    n_samples    = int((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)
    offset_start = int(EPOCH_TMIN * SFREQ)

    # Mapping classi originali -> indici 0-based
    label_map = {1: 0, 2: 1, 3: 2}

    for run in mi_runs:
        X         = run['X'][0, 0]
        y         = run['y'][0, 0].flatten()
        trial     = run['trial'][0, 0].flatten()
        artifacts = run['artifacts'][0, 0].flatten()

        for i in range(len(trial)):
            # Includi solo Left(1), Right(2), Feet(3) — escludi Tongue(4)
            if y[i] in [1, 2, 3] and artifacts[i] == 0:
                start_idx    = trial[i] + offset_start
                end_idx      = start_idx + n_samples
                epoch_signal = X[start_idx:end_idx, :22].T
                mapped_label = label_map[y[i]]

                epochs_data_list.append(epoch_signal)
                events_list.append([current_event_id, 0, mapped_label])
                current_event_id += 1

    if len(epochs_data_list) == 0:
        return None

    epochs_data  = np.array(epochs_data_list) * 1e-6
    events_array = np.array(events_list)

    info = mne.create_info(
        ch_names=EEG_CHANNELS,
        sfreq=SFREQ,
        ch_types=['eeg'] * len(EEG_CHANNELS)
    )

    event_id = {"Left Hand": 0, "Right Hand": 1, "Feet": 2}

    epochs = mne.EpochsArray(
        data=epochs_data,
        info=info,
        events=events_array,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        verbose=False
    )

    epochs.set_montage('standard_1005')
    epochs.apply_baseline(baseline=(EPOCH_TMIN, EPOCH_TMIN + 0.5))

    return epochs


def process_all_subjects():
    print("Inizio conversione BCI IV 2a — 3 classi (Left, Right, Feet):")
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]

    for filename in sorted(mat_files):
        filepath = os.path.join(DATA_DIR, filename)
        epochs   = extract_epochs_3class(filepath)

        if epochs is not None:
            save_name = f"subject_{filename.replace('.mat', '')}_clean-epo.fif"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            epochs.save(save_path, overwrite=True, verbose=False)
            print(f" -> salvato: {save_name} | Epoche valide: {len(epochs)}")


if __name__ == "__main__":
    process_all_subjects()