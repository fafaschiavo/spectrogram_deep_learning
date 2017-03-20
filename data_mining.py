import os
import wave

import pylab
import matplotlib.pyplot as plt

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import urllib2
import subprocess

#ID f8acc77f824346b3bb9b22fbdd200b5e
#SECRET 40e4388e11844b11981f9d3edb597b0f

def graph_spectrogram(wav_file, spectrogram_file_name, spectrogram_folder_name):
    sound_info, frame_rate = get_wav_info(wav_file)
    fig = pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pxx,  freq, t, cax = pylab.specgram(sound_info, Fs=frame_rate)
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
    # print len(sound_info)
    # print frame_rate
    # print number_of_channels
    wav.close()
    return sound_info, frame_rate

def list_genres():
    client_credentials_manager = SpotifyClientCredentials(client_id='f8acc77f824346b3bb9b22fbdd200b5e', client_secret='40e4388e11844b11981f9d3edb597b0f')
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    spotify.trace=False
    genres = spotify.recommendation_genre_seeds()
    return genres

def download_mp3(song_url, file_name, destinatioon_folder):
    # sets the song url to mp3link variable.
    mp3link = song_url

    # opens the .mp3 file - using the same procedure as above.
    openmp3 = open(destinatioon_folder + '/' + file_name + '.mp3', 'w')
    dl = urllib2.urlopen(mp3link)
    dl2 = dl.read()

    # writes the .mp3 file to the file 'testing.mp3' which is in the variable openmp3.
    openmp3.write(dl2)
    openmp3.close()
    return True

def decode_mp3_to_wav(mp3_file_name, mp3_file_folder, wav_file_name, wav_file_folder):

    cmd = 'lame --decode %s/%s.mp3 %s/%s.wav' % (mp3_file_folder, mp3_file_name, wav_file_folder, wav_file_name)
    # p = subprocess.call(cmd, shell=True)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    return wav_file_folder + '/' + wav_file_name + '.wav'



# wav_file = 'track_test5.wav'
# graph_spectrogram(wav_file)

success_counter = 0
request_counter = 0
url_fail_data = 0
file_already_exists = 0
while success_counter < 2:

    client_credentials_manager = SpotifyClientCredentials(client_id='f8acc77f824346b3bb9b22fbdd200b5e', client_secret='40e4388e11844b11981f9d3edb597b0f')
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    spotify.trace=False

    genre_to_mine = 'metal'
    folder_name = genre_to_mine
    genre_string = [folder_name]
    recommendations_genres = spotify.recommendations(seed_genres=genre_string, limit=1)
    request_counter = request_counter + 1
    print "Numero de requisicoes - " + str(request_counter)
    print "Numero de fails URL - " + str(url_fail_data)
    print "Numero de arquivos repetidos evitados - " + str(file_already_exists)

    for track in recommendations_genres['tracks']:
        track_id = track['id']
        track_url = track['preview_url']

        files = os.listdir(folder_name + '/')
        if track_id+'.png' in files:
            new_file = False
            file_already_exists = file_already_exists + 1
        else:
            new_file = True

        if track_url is not None and new_file:
            download_mp3(track_url, track_id, folder_name)
            wav_file = decode_mp3_to_wav(track_id, folder_name, track_id, folder_name)
            graph_spectrogram(wav_file, track_id, folder_name)
            success_counter = success_counter + 1
            print 'Done'
            print 'Success - ' + str(success_counter)
            print '\n'
        else:
            url_fail_data = url_fail_data + 1






