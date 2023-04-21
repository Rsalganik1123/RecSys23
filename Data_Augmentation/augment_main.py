import pandas as pd 
import time 
import ipdb 
import os 
from tqdm import tqdm 
import pickle 
import argparse
import requests
import urllib3
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


from utils.private import * 
from scraping.feature_scrapes import * 
from processing.text_embeddings import * 
from processing.img_embeddings import * 

os.environ['SPOTIPY_CLIENT_ID'] = spotify_client_id
os.environ['SPOTIPY_CLIENT_SECRET'] = spotify_client_secret



def test_spotify_connection(): 
    session = requests.Session()
    retry = urllib3.Retry(
        respect_retry_after_header=False
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(),
                        requests_session=session)

    uri = ['spotify:album:7AUTLy9C6cIsijk9GzOW8T']
    try: 
        results = sp.albums(uri)
    except SpotifyException as e: 
            if e.http_status == 429:
                print("'retry-after' value:", e.headers['retry-after'])
                retry_value = e.headers['retry-after']
                if int(e.headers['retry-after']) > 60: 
                    print("STOP FOR TODAY, retry value too high {}".format(retry_value))
                    

#SCRAPING
def load_mus_feat(data_path, output_path): 
    b = time.time()
    data = pd.read_csv(data_path, delimiter = '\t', quoting = 3)
    a = time.time() 
    print("LOADING TOOK:{}".format(a-b))
    
    uris = data.uri.to_list() 
    scrape_music_feature_data(uris, output_path)

def load_org_info(data_path, output_path): 
    b = time.time()
    data = pd.read_csv(data_path, delimiter = '\t', quoting = 3)
    a = time.time() 
    print("LOADING TOOK:{}".format(a-b))
    uris = data.uri.to_list() 
    scrape_org_info_chunked(uris, output_path)

def load_album_info(data_path, img_path, output_path, offset=0, batch_size=10): 
    data = pickle.load(open(data_path, "rb"))
    uris = data.album_uri.unique().tolist() 
    scrape_album_data_from_album_uri_chunked(uris, img_path, output_path, batch_size = batch_size, offset = offset)

def load_genre_info(data_path, save_path, flag = False): 
    data = pickle.load(open(data_path, "rb"))
    if flag: 
        data = data['df_track']
        uris = data.artist_uri.unique().tolist()
    else: 
        uris = list(set([a[0] for a in data.artist_uris.unique().to_list()]))
    scrape_artist_data_chunked(uris, save_path)
    return 0 

#PRE-PROCESSING 
def gen_text_emb(input_path, output_path, text_key = 'track_name', output_key = 'track_name_emb'): 
    generate_text_features_file(input_path, output_path, text_key, output_key)

def gen_img_emb(data_path, output_path): 
    generate_images_features_file(data_path, output_path, method='resnet')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        help="which function to run",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="data path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        help="path for holding images",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__': 
    args = parse_args()
    print(args)
    if args.mode == 'test': 
        test_spotify_connection()
    if args.mode == 'org': 
        load_org_info(args.input, args.output)
    if args.mode == 'load_mus_feat': 
        load_mus_feat(args.input, args.output)
    if args.mode == "load_album_info": 
        load_album_info(args.input, args.img_path, args.output)
    if args.mode == 'load_genre_info': 
        load_genre_info(args.input, args.output, flag=True)
    if args.mode == 'get_text_emb': 
        gen_text_emb(args.input, args.output)
    if args.mode == 'gen_img_emb': 
        gen_img_emb(args.input, args.output)
    