import argparse
import copy


def command_parser():
    parser = argparse.ArgumentParser()

    subps = parser.add_subparsers()

    '''Adversarial Auto-Encoder'''
    # Used to train the AAE
    subps_aae = subps.add_parser(
        'train',
        help='Train the Adversarial Auto-Encoder')
    # Dataset
    subps_aae.add_argument(
        '--dataset',
        choices=['ZINC', 'ZINCMOSES', 'PCBA'],
        help='Dataset Name'
    )
    # Embed Size
    subps_aae.add_argument(
        "--embed_size",
        type=int, default=32,
        help='Embedding/ Input Size for AAE Encoder and Decoder'
    )
    # Latent Size
    subps_aae.add_argument(
        '--latent_size',
        type=int, default=128,
        help='Size of Latent Vectors'
    )
    # Hidden Size
    subps_aae.add_argument(
        "--hidden_size",
        type=int, default=512,
        help='Size of Hidden Layers'
    )
    # Hidden Layers
    subps_aae.add_argument(
        "--hidden_layers",
        type=int, default=2,
        help='Number of Hidden Layers'
    )
    # Discriminator Layers
    subps_aae.add_argument(
        "--discrim_layers",
        nargs='+', type=int, default=[640, 256],
        help='Number of Features for Discriminator Layers'
    )
    # Multilayer Perceptron Layers
    subps_aae.add_argument(
        "--mlp_layers",
        nargs='+', type=int, default=[640, 256],
        help='Number of Features for Property Predictor Layers'
    )
    # Discriminator Step
    subps_aae.add_argument(
        "--discrim_step",
        type=int, default=3,
        help='Frequency to Train the Discriminator per Auto Encoder Training Step'
    )
    # Dropout
    subps_aae.add_argument(
        "--dropout",
        type=float, default=0.3,
        help='Dropout Probability'
    )
    # Bidirectional
    subps_aae.add_argument(
        "--bidirectional",
        action='store_true',
        help='If included, use Bidirectional LSTM'
    )
    # Use GPU
    subps_aae.add_argument(
        "--use_gpu",
        action='store_true',
        help='Use GPU'
    )
    # Batch Size
    subps_aae.add_argument(
        "--batch_size",
        type=int, default=64,
        help='Batch Size'
    )
    # Epochs
    subps_aae.add_argument(
        "--num_epochs",
        type=int, default=5,
        help='Number of Epochs'
    )
    # Learning Rate
    subps_aae.add_argument(
        "--optim_lr",
        type=float, default=0.0007,
        help='Learning Rate'
    )
    # Seed
    subps_aae.add_argument(
        "--random_seed",
        type=int, default=49,
        help='Random Generator Seed'
    )
    # Scheduler Step Size
    subps_aae.add_argument(
        '--sched_step_size',
        type=int, default=2,
        help='Scheduler Step Size for Training'
    )
    # Scheduler Gamma
    subps_aae.add_argument(
        '--sched_gamma',
        type=float, default=0.9,
        help='Scheduler Gamma Value for Training'
    )
    # Use Scheduler
    subps_aae.add_argument(
        '--use_scheduler',
        action='store_true',
        help='Use Scheduler in Training'
    )
    # Model Name
    subps_aae.add_argument(
        '--model_name',
        type=str, default='saved_AAE_model.pt',
        help='Name to save the model with'
    )

    # Temp Implementation (A3)
    subps_aae.add_argument(
        '--subsets',
        type=int, default=-1,
        help='Number of subsets of the data to use. Min = 2, one set = batch size'
    )

    subps_aae.set_defaults(command='train')

    '''AAE LOAD / SAMPLE'''
    # Used to generate samples from a trained model
    subps_aae_load = subps.add_parser(
        'sample',
        help='Load AAE for Generating Samples'
    )
    # Run Dir
    subps_aae_load.add_argument(
        "--run_dir", metavar="FOLDER",
        help="Directory of the Pre-Trained Adversarial Auto-Encoder"
    )
    # Number of Generated Samples
    subps_aae_load.add_argument(
        "--gen_samples",
        type=int, default=1000,
        help='Number of New Samples to Generate'
    )
    # Max Generated Sample Length
    subps_aae_load.add_argument(
        '--max_len',
        type=int, default=100,
        help='Maximum length of SMILES Streams to be Generated'
    )
    subps_aae_load.set_defaults(command='sample')

    '''Adversarial Auto-Encoder and Deep Evolutionary Learning'''
    # Used for DEL with AAE
    subps_aae_del = subps.add_parser(
        'del',
        help='Deep Evolutionary Learning implemented with Adversarial Auto-Encoder'
    )
    # Dataset
    subps_aae_del.add_argument(
        '--dataset',
        choices=['ZINC', 'PCBA', 'ZINCMOSES'],
        help='Dataset Name'
    )

    # Model Parameters
    # Embed Size
    subps_aae_del.add_argument(
        "--embed_size",
        type=int, default=32,
        help='Embedding/ Input Size for Adversarial Auto-Encoder and Decoder'
    )
    # Latent Size
    subps_aae_del.add_argument(
        '--latent_size',
        type=int, default=128,
        help='Size of Latent Vectors'
    )
    # Hidden Size
    subps_aae_del.add_argument(
        "--hidden_size",
        type=int, default=512,
        help='Size of Hidden Layers'
    )
    # Hidden Layers
    subps_aae_del.add_argument(
        "--hidden_layers",
        type=int, default=2,
        help='Number of Hidden Layers'
    )
    # Discriminator Layers
    subps_aae_del.add_argument(
        "--discrim_layers",
        nargs='+', type=int, default=[640, 256],
        help='Number of Features for Discriminator Layers'
    )
    # Multilayer Perceptron Layers
    subps_aae_del.add_argument(
        "--mlp_layers",
        nargs='+', type=int, default=[640, 256],
        help='Number of Features for Property Predictor Layers'
    )
    # Discriminator Step
    subps_aae_del.add_argument(
        "--discrim_step",
        type=int, default=3,
        help='Frequency to Train the Discriminator per Auto Encoder Training Step'
    )
    # Dropout
    subps_aae_del.add_argument(
        "--dropout",
        type=float, default=0.3,
        help='Dropout Probability'
    )
    # Bidirectional
    subps_aae_del.add_argument(
        "--bidirectional",
        action='store_true',
        help='If included, use Bidirectional LSTM'
    )

    # Training Parameters
    # Use GPU
    subps_aae_del.add_argument(
        "--use_gpu",
        action='store_true',
        help='Use GPU'
    )
    # Batch Size
    subps_aae_del.add_argument(
        "--batch_size",
        type=int, default=64,
        help='Batch Size'
    )
    # Initial Epochs
    subps_aae_del.add_argument(
        "--init_num_epochs",
        type=int, default=20,
        help='Number of Initial Epochs'
    )
    # Learning Rate
    subps_aae_del.add_argument(
        "--optim_lr",
        type=float, default=0.001,
        help='Learning Rate'
    )
    # Seed
    subps_aae_del.add_argument(
        "--random_seed",
        type=int, default=49,
        help='Random Generator Seed'
    )
    # Scheduler Step Size
    subps_aae_del.add_argument(
        '--sched_step_size',
        type=int, default=2,
        help='Scheduler Step Size'
    )
    # Scheduler Gamma
    subps_aae_del.add_argument(
        '--sched_gamma',
        type=float, default=0.9,
        help='Scheduler Gamma Value'
    )
    # Use Scheduler
    subps_aae_del.add_argument(
        '--use_scheduler',
        action='store_true',
        help='Use Scheduler in Training'
    )

    # Evolutionary Parameters
    # Number of Generations
    subps_aae_del.add_argument(
        '--num_generations',
        type=int, default=20,
        help='Number of DEL Generations'
    )
    # Population Size
    subps_aae_del.add_argument(
        '--population_size',
        type=int, default=20000,
        help='Population Size'
    )
    # Subsequent Epochs
    subps_aae_del.add_argument(
        "--subsequent_num_epochs",
        type=int, default=5,
        help='Number of Subsequent Epochs After First Generation'
    )
    # Probability of Tournament Selection
    subps_aae_del.add_argument(
        '--prob_ts',
        type=float, default=0.95,
        help='Probability of Tournament Selection'
    )
    # Crossover Method
    subps_aae_del.add_argument(
        '--crossover',
        type=str, default='linear', choices=['linear', 'discrete'],
        help='Crossover Method'
    )
    # Mutation Rate
    subps_aae_del.add_argument(
        '--mutation',
        type=float, default=0.01,
        help='Mutation Rate'
    )
    # Fine Tune
    subps_aae_del.add_argument(
        '--no_finetune',
        action='store_true',
        help='Do Not Use  Fine Tuning in Subsequent Training'
    )
    subps_aae_del.add_argument(
        '--ranking',
        type=str,
        choices=['sopr', 'fndr'],
        default='fndr',
        help="Type of Ranking Method"

    )
    # Sampling
    # Generate Samples
    subps_aae_del.add_argument(
        '--gen_samples',
        type=int, default=2000,
        help='Number of Samples to Generate from Gauss'
    )

    # Max Length
    subps_aae_del.add_argument(
        '--max_len',
        type=int, default=100,
        help='Max Length of Generated Samples'
    )

    # File Parameters
    # Model Name
    subps_aae_del.add_argument(
        '--model_name',
        type=str, default='saved_AAE_DEL_model.pt',
        help='Name to save the model with'
    )

    # Save Every Population
    subps_aae_del.add_argument(
        '--save_pops',
        default=True,
        action='store_true',
        help='Store Every Population During DEL Training'
    )

    # Temp Implementation (A3)
    subps_aae_del.add_argument(
        '--subsets',
        type=int, default=-1,
        help='Number of subsets of the data to use. Min = 2, one set = batch size'
    )

    subps_aae_del.set_defaults(command='del')

    return parser
