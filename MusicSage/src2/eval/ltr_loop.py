from src2.utils.config.small import get_cfg_defaults
from src2.model.build import build_model
from src2.graph_build.data_load import build_dataset
import src2.graph_build.spotify_dataset
from torch.utils.data import IterableDataset, DataLoader
from src2.sampler.graph_sampler import build_graph_sampler
# from tqdm import tqdm 
from torch.nn.functional import cosine_similarity
from torch.nn import CosineSimilarity
import torch 
import pandas as pd 
import numpy as np 
import pickle
from tqdm import tqdm 
import time 
import numpy as np 
from numpy.linalg import norm 
import os 
import xgboost as xgb



def ltr_train_set(cfg, track_embeddings=None, track_embed_path=None, output_path=None): 
    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, 'rb'))
    df_users = all_data[cfg.DATASET.USER_DF]
    df_interactions = all_data[cfg.DATASET.INTERACTION_DF]
    df_items = all_data[cfg.DATASET.ITEM_DF]
    #load embeddings 
    if track_embed_path: 
        track_embeddings = pickle.load(open(track_embed_path, "rb")) 
    track_embeddings = torch.tensor(track_embeddings)

    train_playlists_sample = df_users.sample(20000, replace=False)
    pids, tids, labels, embs =  [], [], [], [] 

    for pid in tqdm(train_playlists_sample.pid.unique()): 
        associated_tracks = df_interactions[df_interactions.pid == pid]
        unassociated_tracks = df_items.sample(len(associated_tracks), replace=False)
        
        # pids.append([pid]*(len(associated_tracks) + len(unassociated_tracks)))
        # tids.append(associated_tracks.tid.tolist()+unassociated_tracks.tid.tolist())
        # labels.append(associated_tracks.pos.tolist() + [-1]*len(unassociated_tracks))
        # embs.append(track_embeddings[associated_tracks.tid.tolist() + unassociated_tracks.tid.tolist()].numpy())
    
        pids.extend([pid]*(len(associated_tracks) + len(unassociated_tracks)))
        tids.extend(associated_tracks.tid.tolist()+unassociated_tracks.tid.tolist())
        # labels.extend(associated_tracks.pos.tolist() + [-1]*len(unassociated_tracks))
        labels.extend([1]*len(associated_tracks) + [0]*len(unassociated_tracks))
        embs.extend(track_embeddings[associated_tracks.tid.tolist() + unassociated_tracks.tid.tolist()].numpy())
    
    train_set = pd.DataFrame({
        'qid': pids, 
        'tid': tids, 
        'label': labels,
        'emb': embs
    }).sort_values('qid')
    print(len(train_set)) 
    
    pickle.dump(train_set, open(output_path, "wb"))
    return train_set

def ltr_rec_set(cfg, track_embeddings=None, track_embed_path=None, output_path=None): 
    all_data = pickle.load(open(cfg.DATASET.DATA_PATH, 'rb'))
    df_users = all_data[cfg.DATASET.USER_DF]
    df_interactions = all_data[cfg.DATASET.INTERACTION_DF]
    df_items = all_data[cfg.DATASET.ITEM_DF]
    test_set = pickle.load(open(cfg.DATASET.TEST_DATA_PATH, 'rb'))
    if track_embed_path: 
        track_embeddings = pickle.load(open(track_embed_path, "rb")) 
    track_embeddings = torch.tensor(track_embeddings)

    gen_amount = 20 
    k = 10000
    pids, tids, labels, embs =  [], [], [], [] 
    for pid in tqdm(test_set.pid.unique()): 
        associated_tracks = test_set[test_set.pid == pid].sort_values('pos').tid.tolist()
        playlist_embedding = torch.mean(track_embeddings[associated_tracks[:20]], axis=0).reshape(1, -1)
        top = torch.Tensor(cosine_similarity(playlist_embedding, track_embeddings))
        pos_val, pos_idx = torch.topk(top, k)
        # print(pos_val, pos_idx)
        pids.extend([pid]*k)
        tids.extend(pos_idx.numpy()) 
        # labels.extend(range(k))
        labels.extend(pos_val.numpy())
        # embs.extend(track_embeddings[pos_val.numpy()].numpy()) #???? 
        embs.extend(track_embeddings[pos_idx.numpy()].numpy()) 
        
        

    rec_set = pd.DataFrame({
            'qid': pids, 
            'tid': tids, 
            'label': labels,
            'emb': embs
    }).sort_values('qid')
    pickle.dump(rec_set, open('/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/music+genre+meta_focal_norm_contig/recs_TS2_clean/LTR_recset2.pkl', "wb"))
    return rec_set 


def ltr_train(cfg, track_embed_path, train_set=None, rec_set=None, train_set_path=None, rec_set_path=None): 

    if train_set_path: 
        train_set = pickle.load(open(train_set_path, "rb"))
    if rec_set_path: 
        rec_set = pickle.load(open(rec_set_path, "rb"))
    test_set = pickle.load(open(cfg.DATASET.TEST_DATA_PATH, 'rb'))
    track_embeddings = pickle.load(open(track_embed_path, "rb")) 

    train_data= train_set
    X_train = np.array(train_data['emb'].tolist()) #train_data.loc[:, ~train_data.columns.isin(['label'])]
    y_train = train_data.loc[:, train_data.columns.isin(['label'])].to_numpy() 

    test_data= rec_set
    X_test = np.array(test_data['emb'].tolist()) #test_data.loc[:, ~test_data.columns.isin(['label'])]
    y_test = test_data.loc[:, test_data.columns.isin(['label'])].to_numpy() 

    groups = train_data.groupby('qid').size().to_frame('size')['size'].to_numpy() 
    
    model = xgb.XGBRanker(  
        tree_method='gpu_hist',
        booster='gbtree',
        objective='rank:pairwise', #ndcg
        random_state=42, 
        learning_rate=0.1,
        colsample_bytree=0.9, 
        eta=0.05, 
        gamma=1.0,
        max_depth=10, 
        n_estimators=150, 
        subsample=0.75, 
        eval_metric =['ndcg@100'])

    model.fit(X_train, y_train, group=groups, verbose=True)
    overlaps = [] 
    for p in tqdm(rec_set.qid.unique()): 
        associated_tracks = rec_set[rec_set.qid == p].tid.to_list()
        gt = test_set[test_set.pid == p].tid.to_list() 
        options = track_embeddings[associated_tracks]
        pred_idx = np.argsort(model.predict(options))[:len(gt)]
        print(model.predict(gt))
        preds = [associated_tracks[i] for i in pred_idx] 
        print(max(preds), max(gt), len(preds), len(gt))
        overlap =  len(np.intersect1d(preds, gt)) / len(gt)
        print(overlap)
        overlaps.append(overlap)
    print(np.mean(overlaps)) 