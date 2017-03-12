import os
import wave

import pylab

#ID f8acc77f824346b3bb9b22fbdd200b5e
#SECRET 40e4388e11844b11981f9d3edb597b0f

def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('spectrogram5.png')


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    number_of_channels = wav.getnchannels()
    print len(sound_info)
    print frame_rate
    print number_of_channels
    wav.close()
    return sound_info, frame_rate


wav_file = 'track_test5.wav'
graph_spectrogram(wav_file)




