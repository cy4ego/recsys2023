#############################################################################################################

# Set up Sklearn environment
conda create -n sklearn -y scikit-learn python=3.8
conda install -n sklearn -y -c anaconda ipykernel
# /home/humuson/anaconda3/evns/sklearn/bin/python -m ipykernel install --user --name=sklearn
conda install -n sklearn -y -c conda-forge optuna
conda install -n sklearn -y -c conda-forge configargparse
# conda install -n sklearn -y pandas

#############################################################################################################

# Set up GBDT environment
conda create -n gbdt -y  python=3.8
conda install -n gbdt -y -c anaconda ipykernel
# /home/humuson/anaconda3/evns/gbdt/bin/python -m ipykernel install --user --name=gbdt
/home/humuson/anaconda3/evns/gbdt/bin/python -m pip install xgboost==1.5.0
/home/humuson/anaconda3/evns/gbdt/bin/python -m pip install catboost==1.0.3
/home/humuson/anaconda3/evns/gbdt/bin/python -m pip install lightgbm==3.3.1
conda install -n gbdt -y -c conda-forge optuna
conda install -n gbdt -y -c conda-forge configargparse
conda install -n gbdt -y pandas

# # For ModelTrees
/home/humuson/anaconda3/evns/gbdt/bin/python -m pip install https://github.com/schufa-innovationlab/model-trees/archive/master.zip

#############################################################################################################

# Set up Pytorch environment

conda create -n torch -y python=3.8
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia 
# conda install -n torch -y -c anaconda ipykernel
conda install -n torch_v1 -y -c conda-forge optuna
conda install -n torch_v1 -y -c conda-forge configargparse
conda install -n torch_v1 -y scikit-learn
conda install -n torch_v1 -y pandas
conda install -n torch_v1 -y matplotlib
conda install -n torch_v1 -y -c pytorch captum
conda install -n torch_v1 -y shap
# /home/humuson/anaconda3/evns/gbdt/bin/python -m ipykernel install --user --name=torch

conda activate torch_v1

# For TabNet
python -m pip install pytorch-tabnet requests qhoptim lightgbm==3.3.1 einops


#############################################################################################################

# For STG
python -m pip install stg==0.1.2

# For NAM
python -m pip install https://github.com/AmrMKayid/nam/archive/main.zip
python -m pip install tabulate

# For DANet
python -m pip install yacs

#############################################################################################################

conda deactivate torch_v1