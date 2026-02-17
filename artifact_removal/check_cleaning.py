# This file is needed to see the difference before and after the start of the ICA & ICLabel clean.
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_sig_csv
from preprocessing import apply_filters
from artifact_removal import remove_artifacts

# Use a specific file for the task that we are considering (MI for hands)
TEST_FILE = "SUB_001_SIG_02.csv" # Due artefatti
#TEST_FILE = "SUB_063_SIG_06.csv" # Sedici artefatti
PATH_SIG = f"../eegmmidb/{TEST_FILE}"

def visualize():
    print(f"CARICAMENTO {TEST_FILE}...")

    raw = load_sig_csv(PATH_SIG)
    raw = apply_filters(raw)
    data_dirty = raw.get_data() * 1e6 # Conversione di nuovo a uV per il grafico


    # Pulizia
    print("Pulizia in esecuzione...")
    raw_clean, labels_removed = remove_artifacts(raw)
    data_clean = raw_clean.get_data() * 1e6


    # Plotting
    # Per ora con canale frontale (Fp1 -> Sensibile ad occhi e quindi Artefatto EOG)
    # E canale centrale (C3 -> Sensibile alla mano destra)
    ch_eye = raw.ch_names.index('Fp1')
    ch_motor = raw.ch_names.index('C3')

    times = raw.times
    duration = 1000 # Primi 1000 campioni (6 sec)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    # Grafico 1 Canale OCCHI Fp1
    axes[0].plot(times[:duration], data_dirty[ch_eye, :duration], color='red', alpha=0.5, label='Original')
    axes[0].plot(times[:duration], data_clean[ch_eye, :duration], color='black', lw=1, label='Clean (ICA)')
    axes[0].set_title(f"Fp1 channel (eyes) - Blink removal check")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Width (uV)")
    axes[0].legend(["raw data", "clean data"])

    
    # Grafico 2 Canale Motorio Mano Destra C3
    axes[1].plot(times[:duration], data_dirty[ch_motor, :duration], color='red', alpha=0.5, label='Original')
    axes[1].plot(times[:duration], data_clean[ch_motor, :duration], color='black', lw=1, label='Clean (ICA)')
    axes[1].set_title(f"Channel C3 (Motor Cortex) - Signal retention check") # Check su conservazione segnale
    axes[1].set_ylabel("Width (uV)")
    axes[1].set_xlabel("Time (s)") 

    plt.tight_layout()
   # plt.savefig("images/overlay.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    visualize()