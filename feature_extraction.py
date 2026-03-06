import librosa
import numpy as np
import soundfile

def extract_feature(file_name):

    with soundfile.SoundFile(file_name) as sound_file:

        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        mfccs = np.mean(
            librosa.feature.mfcc(
                y=X,
                sr=sample_rate,
                n_mfcc=40
            ).T,
            axis=0
        )

    return mfccs