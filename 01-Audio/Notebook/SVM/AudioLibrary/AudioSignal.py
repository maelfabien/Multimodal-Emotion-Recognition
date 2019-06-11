import os
import numpy
from pydub import AudioSegment
from scipy.fftpack import fft


class AudioSignal(object):

    def __init__(self, sample_rate, signal=None, filename=None):

        # Set sample rate
        self._sample_rate = sample_rate

        if signal is None:

            # Get file name and file extension
            file, file_extension = os.path.splitext(filename)

            # Check if file extension if audio format
            if file_extension in ['.mp3', '.wav']:

                # Read audio file
                self._signal = self.read_audio_file(filename)

            # Check if file extension if video format
            elif file_extension in ['.mp4', '.mkv', 'avi']:

                # Extract audio from video
                new_filename = self.extract_audio_from_video(filename)

                # read audio file from extracted audio file
                self._signal = self.read_audio_file(new_filename)

            # Case file extension is not supported
            else:
                print("Error: file not found or file extension not supported.")

        elif filename is None:

            # Cast signal to array
            self._signal = signal

        else:

            print("Error : argument missing in AudioSignal() constructor.")

    '''
    Function to extract audio from a video
    '''
    def extract_audio_from_video(self, filename):

        # Get video file name and extension
        file, file_extension = os.path.splitext(filename)

        # Extract audio (.wav) from video
        os.system('ffmpeg -i ' + file + file_extension + ' ' + '-ar ' + str(self._sample_rate) + ' ' + file + '.wav')
        print("Sucessfully converted {} into audio!".format(filename))

        # Return audio file name created
        return file + '.wav'

    '''
    Function to read audio file and to return audio samples of a specified WAV file
    '''
    def read_audio_file(self, filename):

        # Get audio signal
        audio_file = AudioSegment.from_file(filename)

        # Resample audio signal
        audio_file = audio_file.set_frame_rate(self._sample_rate)

        # Cast to integer
        if audio_file.sample_width == 2:
            data = numpy.fromstring(audio_file._data, numpy.int16)
        elif audio_file.sample_width == 4:
            data = numpy.fromstring(audio_file._data, numpy.int32)

        # Merge audio channels
        audio_signal = []
        for chn in list(range(audio_file.channels)):
            audio_signal.append(data[chn::audio_file.channels])
        audio_signal = numpy.array(audio_signal).T

        # Flat signals
        if audio_signal.ndim == 2:
            if audio_signal.shape[1] == 1:
                audio_signal = audio_signal.flatten()

        # Convert stereo to mono
        audio_signal = self.stereo_to_mono(audio_signal)

        # Return sample rate and audio signal
        return audio_signal

    '''
    Function to convert an input signal from stereo to mono
    '''
    @staticmethod
    def stereo_to_mono(audio_signal):

        # Check if signal is stereo and convert to mono
        if isinstance(audio_signal, int):
            return -1
        if audio_signal.ndim == 1:
            return audio_signal
        elif audio_signal.ndim == 2:
            if audio_signal.shape[1] == 1:
                return audio_signal.flatten()
            else:
                if audio_signal.shape[1] == 2:
                    return (audio_signal[:, 1] / 2) + (audio_signal[:, 0] / 2)
                else:
                    return -1

    '''
    Function to split the input signal into windows of same size
    '''
    def framing(self, size, step, hamming=False):

        # Rescale windows step and size
        win_size = int(size * self._sample_rate)
        win_step = int(step * self._sample_rate)

        # Number of frames
        nb_frames = 1 + int((len(self._signal) - win_size) / win_step)

        # Build Hamming function
        if hamming is True:
            ham = numpy.hamming(win_size)
        else:
            ham = numpy.ones(win_size)

        # Split signals (and multiply each windows signals by Hamming functions)
        frames = []
        for t in range(nb_frames):
            sub_signal = AudioSignal(self._sample_rate, signal=self._signal[(t * win_step): (t * win_step + win_size)] * ham)
            frames.append(sub_signal)
        return frames

    '''
    Function to compute the magnitude of the Discrete Fourier Transform coefficient
    '''
    def dft(self, norm=False):

        # Commpute the magnitude of the spectrum (and normalize by the number of sample)
        if norm is True:
            dft = abs(fft(self._signal)) / len(self._signal)
        else:
            dft = abs(fft(self._signal))
        return dft

    '''
    Function to apply pre-emphasis filter on signal
    '''
    def pre_emphasis(self, alpha =0.97):

        # Emphasized signal
        emphasized_signal = numpy.append(self._signal[0], self._signal[1:] - alpha * self._signal[:-1])

        return emphasized_signal
