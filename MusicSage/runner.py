import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import torch 
import numpy as np 
import sys 
import os 
import time 

def launch_redress(): 
    dgl.seed(47)
    torch.cuda.manual_seed_all(47)   
    random.seed(47)
    torch.cuda.manual_seed(47)
    np.random.seed(47)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(47)

    
    args_for_function = []  #args_for_pinsage_FAIR_train_LFM[int(task_id)]
    
    if args_for_function.dataset == 'LFM':
        cfg = get_LFM_cfg_defaults()
    

    cfg.MODEL.PINSAGE.PROJECTION.EMB = args_for_function.emb 
    cfg.MODEL.PINSAGE.PROJECTION.ALL_FEATURES = args_for_function.projection_feat 
    cfg.MODEL.PINSAGE.PROJECTION.CONCAT = args_for_function.projection_concat 
    cfg.MODEL.PINSAGE.HIDDEN_SIZE = args_for_function.hidden_size 
    cfg.FAIR.FEAT_SET = args_for_function.fair_feat_set 
    cfg.FAIR.FAIRNESS_BALANCE = args_for_function.gamma
    cfg.FAIR.ALPHA = args_for_function.alpha
    cfg.FAIR.BOOST = args_for_function.boost
    cfg.FAIR.NDCG_METHOD = args_for_function.method
    cfg.FAIR.POP_FEAT = args_for_function.pop_feat
    output_path = args_for_function.output_path

    cfg.OUTPUT_PATH = output_path + "_G_{}_A_{}_B_{}/".format(cfg.FAIR.FAIRNESS_BALANCE, cfg.FAIR.ALPHA, cfg.FAIR.BOOST)

    cfg.FAIR.METHOD = 'REDRESS'

    fair_train_main(cfg) 

    #GENERATE EMBEDDINGS 
    all_checkpoints_path = cfg.OUTPUT_PATH

    #load utility embeddings
    print("***LOADING UTILITY EMBEDDINGS***")
    checkpoint_path, u_epoch = find_latest_checkpoint(all_checkpoints_path, mode='u')
    u_embed_output_path = os.path.join(cfg.OUTPUT_PATH, 'u_track_emb') 
    track_embeddings = gen_track_embeddings(cfg, checkpoint_path, output_path = u_embed_output_path, mode='fullg')
   
    #load fair embeddings 
    print("***LOADING FAIR EMBEDDINGS***")
    checkpoint_path, uf_epoch = find_latest_checkpoint(all_checkpoints_path, mode='u+f')
    uf_embed_output_path = os.path.join(all_checkpoints_path, 'u+f_track_emb') 
    track_embeddings = gen_track_embeddings(cfg, checkpoint_path, output_path = uf_embed_output_path, mode='fullg')


    #GENERATE RECOMMENDATIONS 
    #generate utility embeddings
    print("***GENERATING UTILITY RECOMMENDATIONS***")
    k = cfg.REC.K = 500 #100 
    track_embed_path = cfg.OUTPUT_PATH +'u_track_emb/embeddings_as_array_fullg.pkl'
    u_rec_output_path = os.path.join(cfg.OUTPUT_PATH,  'u_recs_{}_top_{}'.format('TS1', k) ) 
    u_rec_df = gen_recommendations_cosine(cfg, k=k, gen_amount=10, sim_version='cosine', track_embeddings=None, track_embed_path= track_embed_path, output_path=u_rec_output_path) 
    
    #generate fair embeddings
    print("***GENERATING FAIR RECOMMENDATIONS***")
    track_embed_path = cfg.OUTPUT_PATH +'u+f_track_emb/embeddings_as_array_fullg.pkl'
    uf_rec_output_path = os.path.join(cfg.OUTPUT_PATH,  'u+f_recs_{}_top{}'.format('TS1', k))  
    uf_rec_df = gen_recommendations_cosine(cfg, k=k, gen_amount=10, sim_version='cosine', track_embeddings=None, track_embed_path= track_embed_path, output_path=uf_rec_output_path) 

    k = 100 
    rec_file = os.path.join(cfg.OUTPUT_PATH,  u_rec_output_path+'/{}_{}_recommended_tracks.pkl'.format('cosine', 10)) 
    launch_performance_eval(cfg, gen_amount=0, epoch = u_epoch,recommended_track_path = rec_file,  verbose=True, output_path=u_rec_output_path)
    launch_fairness_audit(cfg, k=k, epoch = u_epoch, recommended_track_path=rec_file, output_path=u_rec_output_path, mode='PS')

    rec_file = os.path.join(cfg.OUTPUT_PATH,  uf_rec_output_path+'/{}_{}_recommended_tracks.pkl'.format('cosine', 10)) 
    launch_performance_eval(cfg, gen_amount=0, epoch = uf_epoch, recommended_track_path = rec_file,  verbose=True, output_path=uf_rec_output_path)
    launch_fairness_audit(cfg, k=k, epoch = uf_epoch, recommended_track_path=rec_file, output_path=uf_rec_output_path, mode='PS')

