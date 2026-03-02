import os
import numpy as np
import scipy.io as sio
import mne
from config_bci_iv import SFREQ, EEG_CHANNELS, EPOCH_TMAX, EPOCH_TMIN

DATA_DIR = os.path.join("..", "dataset_bci_iv")
OUTPUT_DIR = os.path.join("..", "data", "clean_bci_iv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_epochs_from_mat(filepath):
    """
    Legge i file .mat del dataset BCI IV 2a e crea un oggetto di tipo MNE Epochs.
    Del file .mat, le prime tre celle sono EOG baseline, mentre dalla 3 alla 8 sono le Run del MI (6 totali).
    """
    print(f"Caricamento {os.path.basename(filepath)}...")

    # Caricamento file MATLAB con scipy
    mat = sio.loadmat(filepath)

    # Presa delle run utili al MI
    mi_runs = mat['data'][0, 3:9]

    epochs_data_list = list()
    events_list = list()
    current_event_id = 0

    # Calcolo dei campioni componenti un'epoca:
    n_samples = int((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)   # Durata campioni = 3 sec => 3 * 250 Hz = 750 campioni ad epoca
    offset_start = int(EPOCH_TMIN * SFREQ) # Da dove iniziare a tagliare

    for run in mi_runs:
        # Estrazione dei dati dalla struttura MATLAB
        X = run['X'][0, 0]                           #
        y = run['y'][0, 0].flatten()                 # Etichette delle classi
        trial = run['trial'][0, 0].flatten()         # Indici temporali di inizio
        artifacts = run['artifacts'][0, 0].flatten() # 0 = pulito; 1 = artefatto

        for i in range(len(trial)):
            # Presa solo delle prove Left Hand e Right Hand (al momento, in caso modificare)
            if y[i] in [1, 2] and artifacts[i] == 0:
                start_idx = trial[i] + offset_start
                end_idx = start_idx + n_samples

                # Presa solo dei primi 22 canali, andando cosi a togliere gli ultimi 3 che sono EOG
                epoch_signal = X[start_idx:end_idx, :22].T

                # Mappatura etichette: 1 = Left va a 0, 2 = Right va a 1
                mapped_label = 0 if y[i] == 1 else 1

                epochs_data_list.append(epoch_signal)

                events_list.append([current_event_id, 0, mapped_label]) # Per MNE
                current_event_id += 1

    # Se nessuna epoca valida è stata trovata -> skip
    if len(epochs_data_list) == 0:
        return None
    
    # Conversione in array con numpy 3D
    epochs_data = np.array(epochs_data_list)
    epochs_data = epochs_data * 1e-6 # COnversione dati da microvolt a Volt

    events_array = np.array(events_list)

    # Creazione info per MNE
    info = mne.create_info(
        ch_names=EEG_CHANNELS,
        sfreq=SFREQ,
        ch_types=['eeg'] * len(EEG_CHANNELS)
    )

    # Costruzione oggetto EpochsArray
    event_id = {"Left Hand": 0, "Right Hand": 1}
    epochs = mne.EpochsArray(
        data=epochs_data,
        info=info,
        events=events_array,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        verbose=False
    )

    epochs.set_montage('standard_1005')
    epochs.apply_baseline(baseline=(None, 0))

    return epochs

def process_all_subjects():
    print("Inizio della conversione del dataset BCI IV:")
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]

    for filename in sorted(mat_files):
        filepath = os.path.join(DATA_DIR, filename)
        epochs = extract_epochs_from_mat(filepath)

        if epochs is not None:
            # Salvataggio del file con tipo .fif
            save_name = f"subject_{filename.replace('.mat','')}_clean-epo.fif"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            epochs.save(save_path, overwrite=True, verbose=False)
            
            print(f" -> salvato: {save_name} (Epoche valide: {len(epochs)})")


if __name__ == "__main__":
    process_all_subjects()