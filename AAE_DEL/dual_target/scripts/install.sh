## create conda env
conda create --name del_aae -y

## activate conda env
conda activate del_aae

# install pip
conda install pip -y

conda install python=3.7 -y

# install conda packages
conda install scipy pandas gensim joblib sh matplotlib seaborn scikit-learn -y

# install pip packages
pip install tensorboardX

# install pytorch
conda install pytorch -c pytorch -y

# install rdkit
conda install rdkit -c rdkit -y

# install botorch
conda install botorch -c pytorch -c gpytorch -c conda-forge
