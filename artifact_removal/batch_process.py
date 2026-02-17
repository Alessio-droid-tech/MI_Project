import os
import mne
import time
from pipeline import process_run
from config import TARGET_RUNS # Controlla che la run SIG del file CSV sia 2 6 o 10

DATASET_PATH = os.path.join("..", "eegmmidb")
OUTPUT_PATH = os.path.join("..", "data", "clean")
os.makedirs(OUTPUT_PATH, exist_ok=True)


def run_batch_processing():
    start_time = time.time()

    subject_ids = range(1, 104) # 103 soggetti nel dataset Mendeley

    print(f"Inizio elaborazione per {len(subject_ids)} potenziali soggetti...")
    print(f"Salvataggio in: {os.path.abspath(OUTPUT_PATH)}")
    print("--" * 30)

    for sub_id in subject_ids:
        sub_str = f"{sub_id:03d}"

        # Lista per accumulare le tre epoche delle 3 run di questo soggetto
        subject_epochs_list = list()

        print(f"\nPROCESSING SUBJECT {sub_str}_{run_str}...")

        # Iterazione sulle run di target (2, 6, 10)
        for run_id in TARGET_RUNS:
            run_str = f"{run_id:02d}"

            sig_file = f"SUB_{sub_str}_SIG_{run_str}.csv"
            ann_file = f"SUB_{sub_str}_ANN_{run_str}.csv"
            
            sig_path = os.path.join(DATASET_PATH, sig_file)
            ann_path = os.path.join(DATASET_PATH, ann_file)  

            # Controlla se i file esistono
            if not os.path.exists(sig_path) or not os.path.exists(ann_path):
                print(f"  Run {run_str} mancante. Skipping.")
                continue

            try:
                # Chiama pipeline che ritorna oggetto MNE Epochs:
                epochs = process_run(sig_path, ann_path)

                # Check sulla presenza di eventi validi nell'epoca
                if len(epochs) > 0:
                    subject_epochs_list.append(epochs)
                    print(f"   Run {run_str}: OK ({len(epochs)} epoche)")
                else:
                    print(f"   Run {run_str}: Nessun evento valido trovato.")
            except Exception as e:
                print(f"   ERRORE su Run {run_str}: {e}")
        

        # FINE DEL SOGGETTO
        
        # Concatenazione delle epoche di questo soggetto (se ne sono state ottenute)
        if len(subject_epochs_list) > 0:
            all_sub_epochs = mne.concatenate_epochs(subject_epochs_list)

            # Salvataggio -> File di tipo .fif (non .npy)
            save_name = f"subject_{sub_str}_clean-epo.fif"
            save_path = os.path.join(OUTPUT_PATH, save_name)

            all_sub_epochs.save(save_path, overwrite=True, verbose=False)
            print(f" >> SALVATO: {save_name} [{len(all_sub_epochs)} epoche totali]")
        else:
            print(f" >> Nessun dato valido per Subject {sub_str}. Skipped.")

    # --- FINE TOTALE ---
    elapsed = time.time() - start_time
    print("\n" + "=" * 30)
    print(f"ELABORAZIONE COMPLETATA in {elapsed/60:.2f} minuti.")

if __name__ == "__main__":
    run_batch_processing()