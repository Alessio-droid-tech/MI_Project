import os
import numpy as np
from pipeline import process_run
from config import TARGET_RUNS # Controlla che la run SIG del file CSV sia 2 6 o 10

DATASET_PATH = os.path.join("..", "eegmmidb")
OUTPUT_PATH = "processed_data"

os.makedirs(OUTPUT_PATH, exist_ok=True)

x_all = []
y_all = []

# Trova tutti i file SIG
sig_files = sorted([
    f for f in os.listdir(DATASET_PATH)
    if "_SIG_" in f and f.endswith(".csv")
])

print(f"Trovati {len(sig_files)} files SIG nel Dataset.")

for sig_file in sig_files:
    parts = sig_file.replace(".csv", "").split("_")
    run_id = int(parts[-1]) # Prende 02, 06 o 10

    if run_id not in TARGET_RUNS: # Saltiamo i file che non compiono task di MI che ci interessano (modificare se si implementano altre funzionalità)
        continue

    print(f"Processing {sig_file} (Target Run: {run_id})...")

    ann_file = sig_file.replace("_SIG_", "_ANN_")

    sig_path = os.path.join(DATASET_PATH, sig_file)
    ann_path = os.path.join(DATASET_PATH, ann_file)

    if not os.path.exists(ann_path):
        print(f"ANN mancante per {sig_file}.")
        continue

    try:
        x, y = process_run(sig_path, ann_path)

        if len(x) > 0:
            x_all.append(x)
            y_all.append(y)
            print(f" -> Aggiunte {len(x)} epoche.")
        else:
            print("Nessun evento trovato in questo file.")

    except Exception as e:
        print(f"ERRORE CRITICO su {sig_file}: {e}")


# Concatenazione finale
if len(x_all) > 0:
    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    np.save(os.path.join(OUTPUT_PATH, "x.npy"), x_all)
    np.save(os.path.join(OUTPUT_PATH, "y.npy"), y_all)

    print("----- ELABORAZIONE COMPLETATA -----")
    print(f"Dataset finale salvato in {OUTPUT_PATH}!")
    print(f"Campioni totali (epoche): {x_all.shape[0]}.")
    print("Dimensioni X:", x_all.shape)
    print("Dimensioni Y:", y_all.shape)
else:
    print("Nessun dato salvato. Eseguire CHECK sui percorsi o sui filtri!")