import os
import wave

import pylab
import matplotlib.pyplot as plt

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import urllib2
import subprocess

def list_genres():
    client_credentials_manager = SpotifyClientCredentials(client_id='f8acc77f824346b3bb9b22fbdd200b5e', client_secret='40e4388e11844b11981f9d3edb597b0f')
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    spotify.trace=False
    genres = spotify.recommendation_genre_seeds()
    return genres

print 'Hi there!'
print 'This are the genres on spotify:'
print list_genres()