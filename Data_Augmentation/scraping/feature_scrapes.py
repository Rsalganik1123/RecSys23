import os
import pickle
import requests
from tqdm import tqdm
from utils.scraping_utils import chunks

import requests
import urllib3
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
import time


def scrape_album_data_from_album_uri_chunked(album_uris, imgs_folder, output_path, batch_size = 10, offset = 0):
    session = requests.Session()
    retry = urllib3.Retry(
        respect_retry_after_header=False
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(),
                        requests_session=session)
    
    batches = list(chunks(album_uris, batch_size))[offset:]
    missed_albums = [] 
    start, end = batch_size*offset, (batch_size*offset) + (batch_size-1) #2730, 2729+30
    for batch in tqdm(batches):
        all_data = []
        path =  output_path + '{}-{}_org.pkl'.format(start, end)
        try:
            time.sleep(8)
            results = sp.albums(batch) #sp.tracks(tracks=batch)
        except SpotifyException as e: 
            if e.http_status == 429:
                print("'retry-after' value:", e.headers['retry-after'])
                retry_value = e.headers['retry-after']
                if int(e.headers['retry-after']) > 60: 
                    print("STOP FOR TODAY, retry value too high {}".format(retry_value))
                    exit() 
                else: 
                    time.sleep(retry_values)
                    missed_albums = missed_albums + batch
                    print("missed track count", len(missed_albums))
                    continue 
            else: 
                print("other Spotify error", e)
                missed_albums = missed_albums + batch
                print("missed track count", len(missed_albums))
                continue 
        except Exception as e:
            print("non-spotify error", e)
            missed_albums = missed_albums + batch
            print("missed track count", len(missed_albums))
            continue
        else:
            for uri, album in zip(batch, results['albums']):
                try:
                    data = {
                        'album_uri': album['uri'],
                        'album_name': album['name'],
                        'album_popularity': album['popularity'],
                        'album_release_date': album['release_date'],
                        'album_total_tracks': album['total_tracks']
                    }
                    imgs = album['images']
                    if len(imgs) > 0:
                        if len(imgs) > 1:
                            idx = 1
                        else:
                            idx = 0
                        img_url = album['images'][idx]['url']
                        img_data = requests.get(img_url).content
                        dl_path = os.path.join(imgs_folder, '{0}.jpg'.format(album['uri']))
                        with open(dl_path, 'wb') as handler:
                            handler.write(img_data)
                        data['image_path'] = dl_path
                    else:
                        data['image_path'] = 'NO_IMAGE'
                    all_data.append(data)
                except:
                    if album is not None:
                        raise
                    missed_albums.append(uri)
        start, end = end+1, end+len(all_data) 
        print("Saving to", path, len(all_data))
        print(all_data[0])   
       
        pickle.dump(all_data, open(path, 'wb')) 
        # break

def scrape_music_features_chunked(track_uris, output_path, batch_size=70, offset=0): 
    session = requests.Session()
    retry = urllib3.Retry(
        respect_retry_after_header=False
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(),
                        requests_session=session)
    
    batches = list(chunks(album_uris, batch_size))[offset:]
    missed_tracks = [] 
    start, end = batch_size*offset, (batch_size*offset) + (batch_size-1)
    for batch in tqdm(batches):
        all_data = []
        path =  output_path + '{}-{}_org.pkl'.format(start, end)
        try:
            time.sleep(8)
            results = sp.audio_features(batch) 
        except SpotifyException as e: 
            if e.http_status == 429:
                print("'retry-after' value:", e.headers['retry-after'])
                retry_value = e.headers['retry-after']
                if int(e.headers['retry-after']) > 60: 
                    print("STOP FOR TODAY, retry value too high {}".format(retry_value))
                    exit() 
                else: 
                    time.sleep(retry_values)
                    missed_tracks = missed_tracks + batch
                    print("missed track count", len(missed_tracks))
                    continue 
            else: 
                print("other Spotify error", e)
                missed_tracks = missed_tracks + batch
                print("missed track count", len(missed_tracks))
                continue 
        except Exception as e:
            print("non-spotify error", e)
            missed_tracks = missed_tracks + batch
            print("missed track count", len(missed_tracks))
            continue
        else:
            for uri, track in zip(batch, results):
                try:
                    data = {
                        'track_uri': uri, 
                        'danceability': track['danceability'],
                        'energy': track['energy'], 
                        'loudness': track['loudness'], 
                        'speechiness': track['speechiness'],
                        'acousticness': track['acousticness'], 
                        'instrumentalness': track['instrumentalness'], 
                        'liveness': track['liveness'], 
                        'valence': track['valence'], 
                        'tempo': track['tempo']
                    }
                    all_data.append(data)
                except:
                    if album is not None:
                        raise
                    missed_tracks.append(uri)
        start, end = end+1, end+len(all_data) 
        print("Saving to", path, len(all_data))
        print(all_data[0])   
       
        pickle.dump(all_data, open(path, 'wb'))

def scrape_artist_data_chunked(artist_uri, output_path, batch_size=40):
    """
    given list of artist uris, load and save artist data
    :param artist_uri:  list of spotify artist uris
    :param save_path:   output path
    :return: metadata of scraped data, missing uris
    """

    session = requests.Session()
    retry = urllib3.Retry(
        respect_retry_after_header=False
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(),
                        requests_session=session)
    
    batches = list(chunks(artist_uri, batch_size))

    missed_artists = [] 
    i = 0 
    start, end = batch_size*i, (batch_size*i) + (batch_size-1) 
    for batch in tqdm(batches):
        all_data = []
        path =  output_path + '{}-{}_org.pkl'.format(start, end)
        try:
            time.sleep(5)
            results = sp.artists(batch)
        except SpotifyException as e: 
            if e.http_status == 429:
                print("'retry-after' value:", e.headers['retry-after'])
                retry_value = e.headers['retry-after']
                if int(e.headers['retry-after']) > 60: 
                    print("STOP FOR TODAY, retry value too high {}".format(retry_value))
                    exit() 
                else: 
                    time.sleep(retry_values)
                    missed_artists = missed_artists + batch
                    continue 
            else: 
                print("other Spotify error", e)
                missed_artists = missed_artists + batch
                continue 
        except Exception as e:
            print("non-spotify error", e)
            missed_artists = missed_artists + batch
            print("missed artist count", len(missed_artists))
            continue 
        else:
            for uri, artist in zip(batch, results['artists']):
                try:
                    data = {
                        'artist_uri': artist['uri'],
                        'artist_name': artist['name'],
                        'popularity': artist['followers']['total'],
                        'genres': artist['genres']
                    }
                    all_data.append(data)
                except:
                    missed_artists.append(uri)
        start, end = end+1, end+len(all_data) 
        print("Saving to", path, len(all_data))
        print(all_data[0])   
        pickle.dump(all_data, open(path, 'wb'))
        