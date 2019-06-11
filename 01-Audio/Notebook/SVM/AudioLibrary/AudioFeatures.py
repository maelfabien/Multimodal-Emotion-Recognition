import numpy
from scipy.fftpack.realtransforms import dct
from scipy.stats import kurtosis, skew
from AudioLibrary.AudioSignal import *


class AudioFeatures:

    def __init__(self, audio_signal, win_size, win_step):

        # Audio Signal
        self._audio_signal = audio_signal

        # Short time features window size
        self._win_size = win_size

        # Short time features window step
        self._win_step = win_step

    '''
    Global statistics features extraction from an audio signals
    '''
    def global_feature_extraction(self, stats=['mean', 'std'], features_list=[], nb_mfcc=12, nb_filter=40, diff=0, hamming=True):

        # Extract short term audio features
        st_features, f_names = self.short_time_feature_extraction(features_list, nb_mfcc, nb_filter, hamming)

        # Number of short term features
        nb_feats = st_features.shape[1]

        # Number of statistics
        nb_stats = len(stats)

        # Global statistics feature names
        feature_names = ["" for x in range(nb_feats * nb_stats)]
        for i in range(nb_feats):
            for j in range(nb_stats):
                feature_names[i + j * nb_feats] = f_names[i] + "_d" + str(diff) + "_" + stats[j]

        # Calculate global statistics features
        features = numpy.zeros((nb_feats * nb_stats))
        for i in range(nb_feats):

            # Get features series
            feat = st_features[:, i]

            # Compute first or second order difference
            if diff > 0:
                feat = feat[diff:] - feat[:-diff]

            # Global statistics
            for j in range(nb_stats):
                features[i + j * nb_feats] = self.compute_statistic(feat, stats[j])

        return features, feature_names

    '''
    Short-time features extraction from an audio signals
    '''
    def short_time_feature_extraction(self, features=[], nb_mfcc=12, nb_filter=40, hamming=True):

        # Copy features list to compute
        features_list = list(features)

        # MFFCs features names
        mfcc_feature_names = []
        if 'mfcc' in features_list:
            mfcc_feature_names = ["mfcc_{0:d}".format(i) for i in range(1, nb_mfcc + 1)]
            features_list.remove('mfcc')

        # Filter banks features names
        fbank_features_names = []
        if 'filter_banks' in features_list:
            fbank_features_names = ["fbank_{0:d}".format(i) for i in range(1, nb_filter + 1)]
            features_list.remove('filter_banks')

        # All Features names
        feature_names = features_list + mfcc_feature_names + fbank_features_names

        # Number of features
        nb_features = len(feature_names)

        # Framming signal
        frames = self._audio_signal.framing(self._win_size, self._win_step, hamming=hamming)

        # Number of frame
        nb_frames = len(frames)

        # Compute features on each frame
        features = numpy.zeros((nb_frames, nb_features))
        cur_pos = 0
        for el in frames:

            # Get signal of the frame
            signal = el._signal

            # Compute the normalize magnitude of the spectrum (Discrete Fourier Transform)
            dft = el.dft(norm=True)

            # Return the first half of the spectrum
            dft = dft[:int((self._win_size * self._audio_signal._sample_rate) / 2)]
            if cur_pos == 0:
                dft_prev = dft

            # Compute features on frame
            for idx, f in enumerate(features_list):
                features[cur_pos, idx] = self.compute_st_features(f, signal, dft, dft_prev,
                                                                  self._audio_signal._sample_rate)

            # Compute MFCCs and Filter Banks
            if len(mfcc_feature_names) > 0:
                features[cur_pos, len(features_list):len(features_list) + len(mfcc_feature_names) + len(fbank_features_names)] = self.mfcc(signal, self._audio_signal._sample_rate,
                                                                   nb_coeff=nb_mfcc, nb_filt=nb_filter, return_fbank=len(fbank_features_names) > 0)
            # Compute Filter Banks
            elif len(fbank_features_names) > 0:
                features[cur_pos, len(features_list) + len(mfcc_feature_names):] = self.filter_banks_coeff(signal, self._audio_signal._sample_rate, nb_filt=nb_filter)

            # Keep previous Discrete Fourier Transform coefficients
            dft_prev = dft
            cur_pos = cur_pos + 1

        return features, feature_names

    '''
    Computes zero crossing rate of a signal
    '''
    @staticmethod
    def zcr(signal):
        zcr = numpy.sum(numpy.abs(numpy.diff(numpy.sign(signal))))
        zcr = zcr / (2 * numpy.float64(len(signal) - 1.0))
        return zcr

    '''
    Computes signal energy of frame
    '''
    @staticmethod
    def energy(signal):
        energy = numpy.sum(signal ** 2) / numpy.float64(len(signal))
        return energy

    '''
    Computes entropy of energy
    '''
    @staticmethod
    def energy_entropy(signal, n_short_blocks=10, eps=10e-8):

        # Total frame energy
        energy = numpy.sum(signal ** 2)
        sub_win_len = int(numpy.floor(len(signal) / n_short_blocks))

        # Length of sub-frame
        if len(signal) != sub_win_len * n_short_blocks:
            signal = signal[0:sub_win_len * n_short_blocks]

        # Get sub windows
        sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

        # Compute normalized sub-frame energies:
        sub_energies = numpy.sum(sub_wins ** 2, axis=0) / (energy + eps)

        # Compute entropy of the normalized sub-frame energies:
        entropy = -numpy.sum(sub_energies * numpy.log2(sub_energies + eps))

        return entropy

    '''
    Computes spectral centroid of frame
    '''
    @staticmethod
    def spectral_centroid_spread(fft, fs, eps=10e-8):

        # Sample range
        sr = (numpy.arange(1, len(fft) + 1)) * (fs / (2.0 * len(fft)))

        # Normalize fft coefficients by the max value
        norm_fft = fft / (fft.max() + eps)

        # Centroid:
        C = numpy.sum(sr * norm_fft) / (numpy.sum(norm_fft) + eps)

        # Spread:
        S = numpy.sqrt(numpy.sum(((sr - C) ** 2) * norm_fft) / (numpy.sum(norm_fft) + eps))

        # Normalize:
        C = C / (fs / 2.0)
        S = S / (fs / 2.0)

        return C, S

    '''
    Computes the spectral flux feature
    '''
    @staticmethod
    def spectral_flux(fft, fft_prev, eps=10e-8):

        # Sum of fft coefficients
        sum_fft = numpy.sum(fft + eps)

        # Sum of previous fft coefficients
        sum_fft_prev = numpy.sum(fft_prev + eps)

        # Compute the spectral flux as the sum of square distances
        flux = numpy.sum((fft / sum_fft - fft_prev / sum_fft_prev) ** 2)

        return flux

    '''
    Computes the spectral roll off
    '''
    @staticmethod
    def spectral_rolloff(fft, c=0.90, eps=10e-8):

        # Total energy
        energy = numpy.sum(fft ** 2)

        # Roll off threshold
        threshold = c * energy

        # Compute cumulative energy
        cum_energy = numpy.cumsum(fft ** 2) + eps

        # Find the spectral roll off as the frequency position
        [roll_off, ] = numpy.nonzero(cum_energy > threshold)

        # Normalize
        if len(roll_off) > 0:
            roll_off = numpy.float64(roll_off[0]) / (float(len(fft)))
        else:
            roll_off = 0.0

        return roll_off

    '''
    Computes the Filter Bank coefficients
    '''
    @staticmethod
    def filter_banks_coeff(signal, sample_rate, nb_filt=40, nb_fft=512):

        # Magnitude of the FFT
        mag_frames = numpy.absolute(numpy.fft.rfft(signal, nb_fft))

        # Power Spectrum
        pow_frames = ((1.0 / nb_fft) * (mag_frames ** 2))
        low_freq_mel = 0

        # Convert Hz to Mel
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))

        # Equally spaced in Mel scale
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nb_filt + 2)

        # Convert Mel to Hz
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))
        bin = numpy.floor((nb_fft + 1) * hz_points / sample_rate)

        # Calculate filter banks
        fbank = numpy.zeros((nb_filt, int(numpy.floor(nb_fft / 2 + 1))))
        for m in range(1, nb_filt + 1):

            # left
            f_m_minus = int(bin[m - 1])

            # center
            f_m = int(bin[m])

            # right
            f_m_plus = int(bin[m + 1])

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(pow_frames, fbank.T)

        # Numerical Stability
        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)

        # dB
        filter_banks = 20 * numpy.log10(filter_banks)

        return filter_banks

    '''
    Computes the MFCCs
    '''
    def mfcc(self, signal, sample_rate, nb_coeff=12, nb_filt=40, nb_fft=512, return_fbank=False):

        # Apply filter bank on spectogram
        filter_banks = self.filter_banks_coeff(signal, sample_rate, nb_filt=nb_filt, nb_fft=nb_fft)

        # Compute MFCC coefficients
        mfcc = dct(filter_banks, type=2, axis=-1, norm='ortho')[1: (nb_coeff + 1)]

        # Return MFFCs and Filter banks coefficients
        if return_fbank is True:
            return numpy.concatenate((mfcc, filter_banks))
        else:
            return mfcc

    '''
    Compute statistics on short time features
    '''
    @staticmethod
    def compute_statistic(seq, statistic):
        if statistic == 'mean':
            S = numpy.mean(seq)
        elif statistic == 'med':
            S = numpy.median(seq)
        elif statistic == 'std':
            S = numpy.std(seq)
        elif statistic == 'kurt':
            S = kurtosis(seq)
        elif statistic == 'skew':
            S = skew(seq)
        elif statistic == 'min':
            S = numpy.min(seq)
        elif statistic == 'max':
            S = numpy.max(seq)
        elif statistic == 'q1':
            S = numpy.percentile(seq, 1)
        elif statistic == 'q99':
            S = numpy.percentile(seq, 99)
        elif statistic == 'range':
            S = numpy.abs(numpy.percentile(seq, 99) - numpy.percentile(seq, 1))
        return S

    '''
    Compute short time features on signal
    '''
    def compute_st_features(self, feature, signal, dft, dft_prev, sample_rate):
        if feature == 'zcr':
            F = self.zcr(signal)
        elif feature == 'energy':
            F = self.energy(signal)
        elif feature == 'energy_entropy':
            F = self.energy_entropy(signal)
        elif feature == 'spectral_centroid':
            [F, FF] = self.spectral_centroid_spread(dft, sample_rate)
        elif feature == 'spectral_spread':
            [FF, F] = self.spectral_centroid_spread(dft, sample_rate)
        elif feature == 'spectral_entropy':
            F = self.energy_entropy(dft)
        elif feature == 'spectral_flux':
            F = self.spectral_flux(dft, dft_prev)
        elif feature == 'sprectral_rolloff':
            F = self.spectral_rolloff(dft)
        return F
