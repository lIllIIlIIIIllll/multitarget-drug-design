from rdkit import rdBase

from aae.aae_DEL import AAE_DEL
from aae.aae_trainer import AAE_Trainer
from learner.dataset2 import SMILESDataset
from utils.config import Config
from utils.parser import command_parser
from utils.postprocess import score_smiles_samples

rdBase.DisableLog('rdApp.*')


def train_aae_model(config: Config):
    # Load the dataset and get SMILES data
    dataset = SMILESDataset(config)
    print(f'Length of Dataset: {len(dataset)}')

    '''Temp Implementation (A1)'''
    subsets = config.get('subsets')
    if subsets > 1:
        end = min((config.get('batch_size') * subsets), len(dataset))
        dataset.data = dataset.get_data()[:end]
    '''End Point'''

    print(f'Number of Data Points: {len(dataset)}')

    '''
        Create the AAE Trainer
        This Creates the Model, Data Loader and Char Vocab
    '''
    trainer = AAE_Trainer(config, dataset.get_vocab())

    '''
        Train and save the model when done
    '''
    save_path = config.path('model') / config.get('model_name')
    trainer.train(dataset.get_loader())
    trainer.save_model(save_path)

    '''
        Generate Samples and Post Process
    '''
    samples = trainer.gen_samples(config)
    '''
        This is more or less the same as load
        Once the model is saved, you can cancel the run at this point
        If you wish to load, generate and post process new data...
        ... you can continue with the "AAE_LOAD" command
    '''
    marks, scores = score_smiles_samples(samples, dataset.get_data())
    j = 0
    for i in range(len(samples)):
        if marks[i]:
            print(samples[i])
            j += 1
    print("Number of Valid Samples:", j)


def aae_del(config: Config):
    print('AAE DEL')
    del1 = AAE_DEL(config)
    del1.train()


if __name__ == '__main__':
    parser = command_parser()
    args = vars(parser.parse_args())
    command = args.pop("command")

    if command == 'train':
        config = Config(args.pop('dataset'), **args)
        print("Training AAE")
        print(config)
        train_aae_model(config)

    elif command == 'del':
        config = Config(args.pop('dataset'), **args)
        print("DEL and AAE")
        print(config)
        aae_del(config)
