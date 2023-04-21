from typing import NamedTuple
import numpy as np


mus_emb_LFM = [
    ['danceability_10cat', 8],
    ['energy_10cat', 8],
    ['loudness_10cat', 8],
    ['speechiness_10cat', 8],
    ['acousticness_10cat', 8],
    ['instrumentalness_10cat', 8],
    ['liveness_10cat', 8],
    ['valence_10cat', 8], 
    ['tempo_10cat', 8],
]

mus_emb_MPD = [['danceability', 16],
    ['energy', 16],
    ['loudness', 16],
    ['speechiness', 16],
    ['acousticness', 16],
    ['instrumentalness', 16],
    ['liveness', 16],
    ['valence', 16], 
    ['tempo', 16]]

all_feat_set_MPD = ['danceability', 'energy', 'loudness',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence',  'img_emb', 'tempo', 'track_name_emb'] #'genres_vec',

all_feat_set_LFM = ['danceability_10cat', 'energy_10cat', 'loudness_10cat',
       'speechiness_10cat', 'acousticness_10cat', 'instrumentalness_10cat', 'liveness_10cat',
       'valence_10cat', 'tempo_10cat',  'img_emb', 'track_name_emb'] #'genres_vec',

mus_feat_LFM = ['tempo_10cat', 'liveness_10cat', 'instrumentalness_10cat', 'speechiness_10cat', 'loudness_10cat', 'acousticness_10cat', 'danceability_10cat', 'valence_10cat', 'energy_10cat']
mus_feat_MPD = ['tempo', 'liveness', 'instrumentalness', 'speechiness', 'loudness', 'acousticness', 'danceability', 'valence', 'energy']

class LaunchArgs(NamedTuple): 
    exp_name: str 
    base_path: str 
    dataset: NamedTuple 
    run_params: NamedTuple
    
class DatasetArgs(NamedTuple): 
    name: str 
    train_path: str 
    test_path: str 

class RedressArgs(NamedTuple):  
    emb: list 
    projection_feat: list 
    dropout: float 
    weight_decay: float 
    projection_concat: list 
    hidden_size: int 
    fair_feat_set: int 
    version: str 
    gamma: float 
    alpha: int 
    boost: float 
    method: str 
    pop_feat: str
    seed: int 
    
class ScoreRegArgs(NamedTuple): 
    lr: float 
    dropout: float 
    batch_size: int 
    epochs: int
    dataset: str 
    top_k: int 
    factor_num: int 
    num_layers: int 
    num_ng: int 
    test_num_ng: int 
    out: bool 
    gpu: str
    version: str 
    sample: str 
    burnin: str 
    weight: float 
    model: str 
    emb_size: int
    reg: str 
    output_path: str
      
class XquadArgs(NamedTuple): 
    gamma: float 
    rec_path: str 
    version: str 
    
versions = dict(zip(['v{}'.format(i) for i in range(1,6)], [56, 83, 24, 2, 4]))

gammas = np.arange(0.1, 1.0, 0.1, dtype=float)

# datasets = {
#     'LFM_Subset': DatasetArgs(name='LFM_Subset', train_path='/home/mila/r/rebecca.salganik/scratch/MusicSAGE_data_final/LFM_Subset/Smaller_Size/train_val_data8.pkl', test_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_data_final/LFM_Subset/Smaller_Size/test_data2.pkl'), 
#     'MPD_Subset': DatasetArgs(name='MPD_Subset', train_path='/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/datasets/small_10000_100/train_val3.pkl', test_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data/datasets/small_10000_100/test2.pkl'),
#     'LFM_Filtered': DatasetArgs(name='LFM_Filtered', train_path='/home/mila/r/rebecca.salganik/scratch/MusicSAGE_data_final/LFM_Filtered/train_val_with_appear2.pkl', test_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_data_final/LFM_Filtered/test.pkl')
#     }

dataset_path = '/home/mila/r/rebecca.salganik/scratch/MusicSAGE_Data_Final2/'
scratch_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/FULL_RUNS' 
exp_path = '/home/mila/r/rebecca.salganik/scratch/PinSAGE_experiments/HP_EXP'

datasets = {
    'LFM_Subset': DatasetArgs(name='LFM_Subset', train_path=f'{dataset_path}LFM_Subset/train_val.pkl', test_path = f'{dataset_path}LFM_Subset/test.pkl'), 
    'MPD_Subset': DatasetArgs(name='MPD_Subset', train_path=f'{dataset_path}MPD_Subset/train_val.pkl', test_path = f'{dataset_path}MPD_Subset/test.pkl'), 
} 

#REDRESS PARAMS 
redress_LFM = [
    RedressArgs(emb=mus_emb_LFM, projection_feat=all_feat_set_LFM, projection_concat=mus_feat_LFM, hidden_size=2048+512+144, fair_feat_set=mus_feat_LFM, alpha=0.01, boost=0.0, dropout = 0.0, weight_decay = 0.0, method='vanilla', pop_feat='log10_popcat', version=v, gamma=float(np.round(g, 2)), seed=s) for  v,s in versions.items()  for g in gammas 
]

redress_MPD = [ 
    RedressArgs(emb=mus_emb_MPD, projection_feat=all_feat_set_MPD, projection_concat=mus_feat_MPD, hidden_size=2048+512+144, fair_feat_set=mus_feat_MPD, alpha=0.01, boost=0.0, dropout = 0.0, weight_decay = 0.0, method='vanilla', pop_feat='log10_popcat', version=v, gamma=float(np.round(g, 2)), seed=s) for  v,s in versions.items()  for g in gammas 
]

#BOOSTING 


#SR PARAMS 
scorereg_MPD = [
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version=v,  weight=np.round(g, 2)) for  v in versions for g in gammas
]

scorereg_LFM = [
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version=v,  weight=np.round(g, 2)) for  v,s in versions.items()  for g in gammas
]
#LGCN_PARAMS
LGCN_MPD = [
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version=v,  weight=0.0) for  v,s in versions.items() 
]

LGCN_LFM = [
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version=v,  weight=0.0) for  v,s in versions.items() 
]

#MACR_PARAMS
 
MACR_MPD = [
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version=v,  weight=0.0) for  v,s in versions.items() 
]

MACR_LFM = [
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version=v,  weight=0.0) for  v,s in versions.items() 
]

#RUNS 
REDRESS_runs_MPD = [
    LaunchArgs(base_path = scratch_path, exp_name='REDRESS', dataset=datasets['MPD_Subset'], run_params=redress_MPD[i]) for i in range(len(redress_MPD))
]

REDRESS_runs_LFM = [
    LaunchArgs(base_path = scratch_path, exp_name='REDRESS', dataset=datasets['LFM_Subset'], run_params=redress_LFM[i]) for i in range(len(redress_LFM))
]

MACR_runs_MPD = [
    LaunchArgs(base_path = scratch_path, exp_name='MACR', dataset=datasets['MPD_Subset'], run_params=MACR_MPD[i]) for i in range(len(MACR_MPD))
]

MACR_runs_LFM = [
    LaunchArgs(base_path = scratch_path, exp_name='MACR2', dataset=datasets['LFM_Subset'], run_params=MACR_LFM [i]) for i in range(len(MACR_LFM))
]

# BOOST_runs_MPD = [
#     LaunchArgs(base_path = scratch_path, exp_name='BOOST', dataset=datasets['MPD_Subset'], run_params=boost_MPD[i]) for i in range(len(boost_MPD))
# ]

# BOOST_runs_LFM = [
#     LaunchArgs(base_path = scratch_path, exp_name='BOOST', dataset=datasets['LFM_Subset'], run_params=boost_LFM[i]) for i in range(len(boost_LFM))
# ]

SR_runs_MPD = [
    LaunchArgs(base_path = scratch_path, exp_name='SR2', dataset=datasets['MPD_Subset'], run_params=scorereg_MPD[i]) for i in range(len(scorereg_MPD))
]

SR_runs_LFM = [
    LaunchArgs(base_path = scratch_path, exp_name='SR2', dataset=datasets['LFM_Subset'], run_params=scorereg_LFM[i]) for i in range(len(scorereg_LFM))
]

LGCN_runs_MPD = [
    LaunchArgs(base_path = scratch_path, exp_name='LGCN2', dataset=datasets['MPD_Subset'], run_params=LGCN_MPD[i]) for i in range(len(LGCN_MPD))
]

LGCN_runs_LFM = [
    LaunchArgs(base_path = scratch_path, exp_name='LGCN2', dataset=datasets['LFM_Subset'], run_params=LGCN_LFM[i]) for i in range(len(LGCN_LFM))
]

XQUAD_runs_MPD = [
    LaunchArgs(base_path = scratch_path, exp_name='XQUAD', dataset=datasets['MPD_Subset'], run_params=redress_MPD[i]) for i in range(len(redress_MPD))
]

XQUAD_runs_LFM = [
    LaunchArgs(base_path = scratch_path, exp_name='XQUAD', dataset=datasets['LFM_Subset'], run_params=redress_LFM[i]) for i in range(len(redress_LFM))
]


#HYPERPARAMS 
emb_hp = [
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 24, model='LightGCN', reg='no', version='v1',  weight=0.0), 
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 64, model='LightGCN', reg='no', version='v1',  weight=0.0), 
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 128, model='LightGCN', reg='no', version='v1',  weight=0.0)
    ] 

emb_size_hp_runs_LFM = [
    LaunchArgs(base_path = exp_path, exp_name = 'LGCN_EMB', dataset=datasets['LFM_Subset'], run_params = emb_hp[i]) for i in range(len(emb_hp)) 
    ]

lr_hp = [
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version='v1',  weight=0.0), 
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version='v1',  weight=0.0), 
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.01, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version='v1',  weight=0.0)
    ] 

lr_hp_runs_LFM = [
    LaunchArgs(base_path = exp_path, exp_name = 'LGCN_LR', dataset=datasets['LFM_Subset'], run_params = lr_hp[i]) for i in range(len(lr_hp)) 
    ]

d2_test = [
    LaunchArgs(base_path = exp_path, exp_name='LGCN', dataset=datasets['MPD_Subset'], run_params=LGCN_MPD[i]) for i in range(len(LGCN_MPD))
]

#BOOSTING TESTS 
boost_methods = ['boost2','boost3','boost4','boost5']
boost_MPD =  [
    # RedressArgs(emb=mus_emb_MPD, projection_feat=all_feat_set_MPD, projection_concat=mus_feat_MPD, hidden_size=2048+512+144, fair_feat_set=mus_feat_MPD, alpha=0.01, pop_feat='log10_popcat', dropout = 0.0, weight_decay = 0.0, version='v1', gamma=0.5, boost = 0.0, method='boost2'), 
    # RedressArgs(emb=mus_emb_MPD, projection_feat=all_feat_set_MPD, projection_concat=mus_feat_MPD, hidden_size=2048+512+144, fair_feat_set=mus_feat_MPD, alpha=0.01, pop_feat='log10_popcat', dropout = 0.0, weight_decay = 0.0, version='v1', gamma=0.5, boost = 0.0, method='boost3'), 
    # RedressArgs(emb=mus_emb_MPD, projection_feat=all_feat_set_MPD, projection_concat=mus_feat_MPD, hidden_size=2048+512+144, fair_feat_set=mus_feat_MPD, alpha=0.01, pop_feat='log10_popcat', dropout = 0.0, weight_decay = 0.0, version='v1', gamma=0.5, boost = 0.0, method='boost4'), 
    RedressArgs(emb=mus_emb_MPD, projection_feat=all_feat_set_MPD, projection_concat=mus_feat_MPD, hidden_size=2048+512+144, fair_feat_set=mus_feat_MPD, alpha=0.01, pop_feat='log10_popcat', dropout = 0.0, weight_decay = 0.0, boost = 0.0, version=v, gamma=float(np.round(g, 2)), method='boost2', seed=47)  for v,s in versions.items() for g in gammas 
]

boost_LFM = [
    # RedressArgs(emb=mus_emb_LFM, projection_feat=all_feat_set_LFM, projection_concat=mus_feat_LFM, hidden_size=2048+512+144, fair_feat_set=mus_feat_LFM, alpha=0.01, boost=0.0, dropout = 0.0, weight_decay = 0.0, method='boost2', pop_feat='log10_popcat', version='v1', gamma=0.5),  
    # RedressArgs(emb=mus_emb_LFM, projection_feat=all_feat_set_LFM, projection_concat=mus_feat_LFM, hidden_size=2048+512+144, fair_feat_set=mus_feat_LFM, alpha=0.01, boost=0.0, dropout = 0.0, weight_decay = 0.0, method='boost3', pop_feat='log10_popcat', version='v1', gamma=0.5),  
    # RedressArgs(emb=mus_emb_LFM, projection_feat=all_feat_set_LFM, projection_concat=mus_feat_LFM, hidden_size=2048+512+144, fair_feat_set=mus_feat_LFM, alpha=0.01, boost=0.0, dropout = 0.0, weight_decay = 0.0, method='boost4', pop_feat='log10_popcat', version='v1', gamma=0.5), 
    RedressArgs(emb=mus_emb_LFM, projection_feat=all_feat_set_LFM, projection_concat=mus_feat_LFM, hidden_size=2048+512+144, fair_feat_set=mus_feat_LFM, alpha=0.01, boost=0.0, dropout = 0.0, weight_decay = 0.0, pop_feat='log10_popcat', version=v, gamma=float(np.round(g, 2)), method='boost2', seed=47)  for  v,s in versions.items()  for g in gammas 
]


BOOST_hp_runs_MPD = [
    LaunchArgs(base_path = scratch_path, exp_name='BOOST', dataset=datasets['MPD_Subset'], run_params=boost_MPD[i]) for i in range(len(boost_MPD))
]

BOOST_hp_runs_LFM = [
    LaunchArgs(base_path = scratch_path, exp_name='BOOST', dataset=datasets['LFM_Subset'], run_params=boost_LFM[i]) for i in range(len(boost_LFM))
]


tiny_redress = [
    RedressArgs(emb=mus_emb_MPD, projection_feat=all_feat_set_MPD, projection_concat=mus_feat_MPD, hidden_size=2048+512+144, fair_feat_set=mus_feat_MPD, alpha=0.01, boost=0.0, dropout = 0.0, weight_decay = 0.0, method='vanilla', pop_feat='log10_popcat', version='v1', gamma=float(1.0), seed=47) 
]
tiny_redress_run = [LaunchArgs(base_path = scratch_path, exp_name='CF_BOOST', dataset='MANUAL', run_params=tiny_redress[0])]

tiny_boost = [ 
    RedressArgs(emb=mus_emb_MPD, projection_feat=all_feat_set_MPD, projection_concat=mus_feat_MPD, hidden_size=2048+512+144, fair_feat_set=mus_feat_MPD, alpha=0.01, boost=0.0, dropout = 0.0, weight_decay = 0.0, method='boost2', pop_feat='log10_popcat', version='v1', gamma=float(1.0), seed=47) 
]
tiny_boost_run = [LaunchArgs(base_path = scratch_path, exp_name='CF_BOOST', dataset='MANUAL', run_params=tiny_boost[0])]

tiny_macr = [
    ScoreRegArgs(output_path = '', dataset='mus', lr = 0.001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='macr', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version='v1',  weight=0.0) 
]

tiny_macr_run = [
    LaunchArgs(base_path = scratch_path, exp_name='CF_MACR', dataset='MANUAL', run_params=tiny_macr[0])
]

# dropout_LFM = [
#     RedressArgs(emb=mus_emb_LFM, projection_feat=all_feat_set_LFM, projection_concat=mus_feat_LFM, hidden_size=2048+512+144, fair_feat_set=mus_feat_LFM, alpha=0.01, boost=0.0, dropout = 0.0, weight_decay = 0.0, method='vanilla', pop_feat='log10_popcat', version="DROPOUT_0.0", gamma=0.1), 
#     RedressArgs(emb=mus_emb_LFM, projection_feat=all_feat_set_LFM, projection_concat=mus_feat_LFM, hidden_size=2048+512+144, fair_feat_set=mus_feat_LFM, alpha=0.01, boost=0.0, dropout = 0.3, weight_decay = 0.0, method='vanilla', pop_feat='log10_popcat', version="DROPOUT_0.3", gamma=0.1), 
#     RedressArgs(emb=mus_emb_LFM, projection_feat=all_feat_set_LFM, projection_concat=mus_feat_LFM, hidden_size=2048+512+144, fair_feat_set=mus_feat_LFM, alpha=0.01, boost=0.0, dropout = 0.5, weight_decay = 0.0, method='vanilla', pop_feat='log10_popcat', version="DROPOUT_0.5", gamma=0.1), 
#     RedressArgs(emb=mus_emb_LFM, projection_feat=all_feat_set_LFM, projection_concat=mus_feat_LFM, hidden_size=2048+512+144, fair_feat_set=mus_feat_LFM, alpha=0.01, boost=0.0, dropout = 0.7, weight_decay = 0.0, method='vanilla', pop_feat='log10_popcat', version="DROPOUT_0.7", gamma=0.1), 
# ]

# dropout_runs_LFM = [
#     LaunchArgs(base_path = exp_path, exp_name='DROPOUT', dataset=datasets['LFM_Subset'], run_params=dropout_LFM[i]) for i in range(len(dropout_LFM))
# ]



# redress_LFM_Filtered = [
#     RedressArgs(emb=mus_emb_LFM, projection_feat=all_feat_set_LFM, projection_concat=mus_feat_LFM, hidden_size=2048+512+144, fair_feat_set=mus_feat_LFM, alpha=0.01, boost=0.0, method='vanilla', pop_feat='80_20_LT', version=v, gamma=float(np.round(g, 2))) for v in versions for g in gammas 
# ]



# redress_MPD = [ 
#     RedressArgs(emb=mus_emb_MPD, projection_feat=all_feat_set_MPD, projection_concat=mus_feat_MPD, hidden_size=2048+512+144, fair_feat_set=mus_feat_MPD, version=v, gamma=float(np.round(g, 2)), alpha=0.01, boost=0.0, method='vanilla', pop_feat='log10_popcat') for v in versions for g in gammas 
# ]

# boost_MPD = [ 
#     RedressArgs(emb=mus_emb_MPD, projection_feat=all_feat_set_MPD, projection_concat=mus_feat_MPD, hidden_size=2048+512+144, fair_feat_set=mus_feat_MPD, version=v, gamma=float(np.round(g, 2)), alpha=0.01, boost=0.0, method='weighted2', pop_feat='log10_popcat') for v in versions for g in gammas 
# ]

# scorereg_MPD = [
#     ScoreRegArgs(output_path = '', dataset='mus', lr = 0.001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1",  burnin="no", emb_size = 10, model='NGCF', reg='no', version=v,  weight=np.round(g, 2)) for  v in versions for g in gammas
# ]

# scorereg_LFM_Filtered = [
#     ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='posneg', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version=v,  weight=np.round(g, 2)) for  v in versions for g in gammas
# ] 

# NGCF_LFM = [
#     ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 10, model='NGCF', reg='no', version=v,  weight=0.0) for  v in versions
# ]

# NGCF_LFM_Filtered = [
#     ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version=v,  weight=0.0) for  v in versions
# ]





# NGCF_runs_LFM = [
#     LaunchArgs(exp_name = 'NGCF', dataset = datasets['LFM'], run_params = NGCF_LFM[i]) for i in range(len(NGCF_LFM))
# ]

# NGCF_runs_LFM_Filtered = [
#     LaunchArgs(exp_name = 'LGCN', dataset = datasets['LFM_Filtered'], run_params = NGCF_LFM_Filtered[i]) for i in range(len(NGCF_LFM_Filtered))
# ]

# SR_runs_LFM_Filtered = [
#     LaunchArgs(exp_name = 'SR', dataset = datasets['LFM_Filtered'], run_params = scorereg_LFM_Filtered [i]) for i in range(len(scorereg_LFM_Filtered))
# ]


# SR_LFM_Filtered = [
#     ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=500, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  burnin="no", emb_size = 10, model='LightGCN', reg='no', version=v,  weight=0.0) for  v in versions
# ]



# BOOST_runs_MPD = [
#     LaunchArgs(exp_name='BOOSTING', dataset=datasets['MPD'], run_params=redress_MPD[i]) for i in range(len(redress_MPD))
# ]

# REDRESS_runs_LFM = [
#     LaunchArgs(exp_name='REDRESS', dataset=datasets['LFM'], run_params=redress_LFM[i]) for i in range(len(redress_LFM))
# ]

# REDRESS_runs_LFM_Filtered = [
#     LaunchArgs(exp_name='REDRESS', dataset=datasets['LFM_Filtered'], run_params=redress_LFM[i]) for i in range(len(redress_LFM_Filtered))
# ]

# PS_runs_LFM_Filtered = [
#     LaunchArgs(exp_name = 'PS', dataset = datasets['LFM_Filtered'], run_params =  redress_LFM_Filtered[i]) for i in range(len(redress_LFM_Filtered))
# ]
# BOOST_runs_LFM = [
#     LaunchArgs(exp_name='BOOSTING', dataset=datasets['LFM'], run_params=redress_LFM[i]) for i in range(len(redress_LFM))
# ]

# SR_runs = [
#     LaunchArgs(exp_name='SR', dataset=datasets['LFM'], run_params=scorereg_MPD[i]) for i in range(len(scorereg_MPD))
# ]

# XQUAD_runs_LFM = [
#     LaunchArgs(exp_name='XQUAD',dataset=datasets['LFM'], run_params = redress_LFM[i]) for i in range(len(redress_LFM))
# ] 

# XQUAD_runs_LFM_Filtered = [
#     LaunchArgs(exp_name='XQUAD',dataset=datasets['LFM_Filtered'], run_params = redress_LFM_Filtered[i]) for i in range(len(redress_LFM_Filtered))
# ] 

# SR_test = LaunchArgs(exp_name='SR', dataset=datasets['MPD'], run_params=ScoreRegArgs(output_path = '', dataset='mus', lr = 0.0001, dropout = 0.0, batch_size=256, epochs=20, top_k =10, factor_num=64, num_layers=3, num_ng=3, test_num_ng=99, sample='none', out=True, gpu="1",  
#     burnin="no", emb_size = 10, model='NGCF', reg='no', version='v1',  weight=0.1)) 



# class Test(NamedTuple): 
#     name: str 
#     version:str 
#     gamma: float 

# Args =  [
#     Test(name='hi', version=i, gamma=np.round(g, 2)) for i in versions for g in gammas] 



