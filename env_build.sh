#!/bin/bash
#SBATCH --job-name=dgx_setup_debugging
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:15:00              # should setup in under 15 minutes, right?

set -e
module load miniconda/3
module load cuda/11.1
# conda create -y -p $SLURM_TMPDIR/env python=3.9
# conda activate $SLURM_TMPDIR/env
conda create --name dgl_1.0_v3 python=3.9
conda activate dgl_1.0_v3 

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --pre dgl -f https://data.dgl.ai/wheels/cu113/repo.html
pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
echo "cuda visible devices: $CUDA_VISIBLE_DEVICES"
unset CUDA_VISIBLE_DEVICES
echo "Checking if torch + cuda works."
python -c 'import torch;assert torch.cuda.is_available() and torch.rand(10, device="cuda").device.type == "cuda"'
echo "Success"
echo "Checking if DGL + cuda works:"
python -c 'import dgl;import torch;src_ids = torch.tensor([2, 3, 4]);dst_ids = torch.tensor([1, 2, 3]);g = dgl.graph((src_ids, dst_ids));g = g.to("cuda"); assert g.device.type == "cuda"'
echo "success"
echo "Checking if DGL seeding works:"
python -c 'import dgl; dgl.seed(45)'
echo "success"
# conda env export -p $SLURM_TMPDIR/env > ./dgl_env_$SLURM_JOB_ID.yml #~/dgl_debugging/dgl_env_$SLURM_JOB_ID.yml
conda env export -p dgl_1.0_v3 > ./dgl_env_$SLURM_JOB_ID.yml

# set -e
# module load miniconda/3
# conda create -y -p $SLURM_TMPDIR/env python=3.9
# conda activate $SLURM_TMPDIR/env

# ## Conda version
# # conda install -y pytorch=12.1 pytorch-cuda=11.6 -c pytorch -c nvidia
# conda install pytorch==1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# # conda install -y spotipy -c conda-forge
# # conda install pyg -c pyg
# conda install -y -c dglteam/label/cu116 dgl


# # Other packages that would need to be installed:
# # pip install pandas scikit-learn fvcore  iopath  numpy scipy spotipy tqdm  wandb

# python -c 'print("torch+cuda works?"); import torch;assert torch.rand(10, device="cuda").device.type == "cuda"'
# python -c 'print("DGL seeding works?"); import dgl; dgl.seed(45)'
# python -c 'print("DGL + uda works?"); import dgl;import torch;src_ids = torch.tensor([2, 3, 4]);dst_ids = torch.tensor([1, 2, 3]);g = dgl.graph((src_ids, dst_ids));g = g.to("cuda"); print(g.device)'

# conda env export -p $SLURM_TMPDIR/env > ~/Projects/dgl_env_$SLURM_JOB_ID.yml