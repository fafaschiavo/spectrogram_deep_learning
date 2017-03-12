import os
import wave

import pylab

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

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

def list_genres():
    client_credentials_manager = SpotifyClientCredentials(client_id='f8acc77f824346b3bb9b22fbdd200b5e', client_secret='40e4388e11844b11981f9d3edb597b0f')
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    spotify.trace=False
    genres = spotify.recommendation_genre_seeds()
    return genres

# wav_file = 'track_test5.wav'
# graph_spectrogram(wav_file)






client_credentials_manager = SpotifyClientCredentials(client_id='f8acc77f824346b3bb9b22fbdd200b5e', client_secret='40e4388e11844b11981f9d3edb597b0f')
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
spotify.trace=False

# name = 'Chroma Key'
# results = spotify.search(q='artist:' + name, type='artist')
# print results

genre_string = ['black-metal']
recommendations_genres = spotify.recommendations(seed_genres=genre_string, limit=100)
print recommendations_genres[0]








