# Code for the paper "Adversarial Deep Evolutionary Learning for Drug Design" (Revised CIBCB 2022)

### Installation

These instructions are based on a Linux based Operating System. 
Ubuntu was primarily used in the development of this project.
Some runs were done on Windows toward the final stages, as such, there are commented duplicate code blocks to account for the different systems.

Requirements
The only requirement for this project is the latest Conda package manager.
This can be downloaded from the Anaconda Python distribution [here](https://www.anaconda.com/distribution).

Run:

`source scripts/install.sh`

This installation will create and activate a new conda venv named `del_aae`, as well as install all required dependencies in this venv. 

If you would like to install these dependencies on a pre-existing environment, please comment out the first two commands (the lines under the comments marked with a two `#`).
That is, `conda create --name del_aae -y` becomes `#conda create --name del_aae -y` and `conda activate del_aae` becomes `#conda activate del_aae`


If you have trouble during the installation, try running each line of the `scripts/install.sh` file separately (one by one) in your shell.

After that, you are all set up.


### Training

You can train the model running:

`python manage.py del --dataset <DATASET_NAME>`

where `<DATASET_NAME>` is defined as described above.
By default, samples are saved every first, half and final generation. 
If you wish to save every sample from every generation, add the `--save_pops` option.

If you wish to train using a GPU, add the `--use_gpu` option.


Check out `python manage.py del --help` to see all the other hyperparameters you can change.

Training the model will create folder `RUNS` with the following structure:

```
RUNS
└── <date>@<time>-<hostname>-<dataset>
    ├── ckpt
    ├── code
    ├── config
    │   ├── config.pkl
    │   └── params.json
    ├── model
    │   └── del_pretrain.pt
    ├── results
    │   ├── bo
    │   ├── performance
    │   │   ├── running_time.csv
    │   │   ├── vnds_dgm.csv
    │   │   └── vnds_pop.csv
    │   ├── samples
    │   └── samples_del
    └── tb
```


the `<date>@<time>-<hostname>-<dataset>` folder is a snapshot of your experiment, which will contain all the data collected during training.
The `code` folder contains a snapshot of the code used to run the experiment and `samples_del` folder contains the samples used in the experiment.