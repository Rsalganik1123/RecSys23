# RecSys23
This code is intended for the RecSys 2023 submission. 

There are two major folders: Data Augmentation and MusicSAGE. 

# Data Augmentation 
***Note: in order to use the spotipy library you have to go on [this link](https://developer.spotify.com/dashboard/) and register for a developer token and secret key. Once you have those, you can add them to the ```utils/private.py``` file. 

There are three steps which need to be done in order to reproduce our datasets:
1. First you can download the respective datasets from their online databases: [MPD](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files) and [LFM2B](http://www.cp.jku.at/datasets/LFM-2b/). 
2. Launch the scraping from the Spotify API. 
    - URIs (only needed for LFM2B): ```python augment_main.py --mode org --input <input_path> --output <output_path.pkl>```
    - Music features: ```python augment_main.py --mode load_mus_feat --input <input_path> --output <output_path.pkl>```
    - Album features: ```python augment_main.py --mode load_album_info --input <input_path> --output <output_path.pkl>```
    - Artist features: ```python augment_main.py --mode load_genre_info --input <input_path> --output <output_path.pkl>```
3. Launch the preprocessing for image and track names: 
    - Album artwork: ```python augment_main.py --mode gen_img_emb --input <input_path> --output <output_path.pkl>```
    - Track names: ```python augment_main.py --mode gen_text_emb--input <input_path> --output <output_path.pkl>```


