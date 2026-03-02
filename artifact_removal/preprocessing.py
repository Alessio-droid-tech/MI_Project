from config import LOW_FREQ, HIGH_FREQ, NOTCH_FREQ

def apply_filters(raw):
    """
    Applica filtri passa-banda e notch al segnale EEG.
    """
    # Filtro passa-banda
    raw.filter(
        l_freq=LOW_FREQ, # Filtro basso ad 1 Hz
        h_freq=HIGH_FREQ, # Filtro alto a 40 Hz
        fir_design='firwin' # FIR design -> Uses scipy.signal.firwin function -> Window Method
    )
    
    # Filtro notch
    raw.notch_filter(
        freqs=NOTCH_FREQ # Applica filtro di Notch a 60 Hz (America)
    )
    
    return raw