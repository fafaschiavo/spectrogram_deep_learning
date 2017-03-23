import os
import wave
import time
import pylab

#ID f8acc77f824346b3bb9b22fbdd200b5e
#SECRET 40e4388e11844b11981f9d3edb597b0f

start_time = time.time()

def graph_spectrogram(wav_file, spectrogram_file_name, spectrogram_folder_name):
    sound_info, frame_rate = get_wav_info(wav_file)
    fig = pylab.figure(num=None, figsize=(19, 12), dpi=500)
    my_subplot = pylab.subplot(111)
    my_subplot.set_yscale('symlog')
    my_subplot.set_ylim(bottom=20, top=10000)
    pylab.title('spectrogram of %r' % wav_file)
    pxx,  freq, t, cax = pylab.specgram(sound_info, Fs=frame_rate, NFFT=8192) #NFFT 4096
    fig.colorbar(cax).set_label('Intensity [dB]')
    pylab.savefig(spectrogram_folder_name + '/' + spectrogram_file_name + '.png')
    pylab.close('all')

    # plt.specgram(sound_info, Fs=frame_rate)
    # plt.savefig(spectrogram_folder_name + '/' + spectrogram_file_name + '-matplotlib.png')
    # plt.close('all')


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    number_of_channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    print len(sound_info)
    print frame_rate
    print number_of_channels
    print sample_width
    wav.close()
    # return sound_info[1000000:1160000], frame_rate
    return sound_info, frame_rate


wav_file = 'samba/3JAnxdVlMLo27vasMzMdPk.wav'
# wav_file = 'samba/metal.wav'
graph_spectrogram(wav_file, 'samba_tuning_spectrogram2', 'samba')

print("--- %s seconds ---" % (time.time() - start_time))


