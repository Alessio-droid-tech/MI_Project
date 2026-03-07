import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt


class FilterBankRiemannian(BaseEstimator, TransformerMixin):
    """
    Invece di usare solo la banda 8-30, esegue un controllo TangentSpace su più bande di frequenza
    e concatena poi i risultati delle bande (Filter Bank RIemannian).
    """
    def __init__(self, freq_bands=None, sfreq=250, estimator='wlf'):
        self.freq_bands = freq_bands or [
            (4, 8),    # Nuova, utilizzare solo per i piedi
            (8, 12),   # Mu
            (12, 16),  # Beta bassa
            (16, 24),  # Beta media
            (24, 32),  # Beta alta
        ]

        self.sfreq = sfreq
        self.estimator = estimator


    def _bandpass(self, X, low, high):
        sos = butter(5, [low, high], btype='bandpass', fs=self.sfreq, output='sos')
        return sosfiltfilt(sos, X, axis=-1)
    

    def fit(self, X, y=None):
        self._cov_list = list()
        self._ts_list = list()

        for (low, high) in self.freq_bands:
            X_filt = self._bandpass(X, low, high)
            cov = Covariances(estimator=self.estimator)
            C = cov.fit_transform(X_filt, y)
            ts = TangentSpace()
            ts.fit(C, y)
            self._cov_list.append(cov)
            self._ts_list.append(ts)

        return self


    def transform(self, X, y=None):
        features = list()
        for i, (low, high) in enumerate(self.freq_bands):
            X_filt = self._bandpass(X, low, high)
            C = self._cov_list[i].transform(X_filt)
            feat = self._ts_list[i].transform(C)
            features.append(feat)
        return np.concatenate(features, axis=1)