from nnsight import LanguageModel
from buffer import ActivationBuffer
from training import trainSAE
from trainers.standard import StandardTrainer
from dictionary import AutoEncoder

model = LanguageModel(
    'EleutherAI/pythia-70m-deduped', # this can be any Huggingface model
    device_map = 'cuda:0'
)
submodule = model.gpt_neox.layers[1].mlp # layer 1 MLP
activation_dim = 512 # output dimension of the MLP
dictionary_size = 16 * activation_dim

# data much be an iterator that outputs strings
data = iter([
    'This is some example data',
    'In real life, for training a dictionary',
    'you would need much more data than this'
])
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    d_submodule=activation_dim, # output dimension of the model component
    n_ctxs=3e4, # you can set this higher or lower dependong on your available memory
    device='cuda:0' # doesn't have to be the same device that you train your autoencoder on
) # buffer will return batches of tensors of dimension = submodule's output dimension

# train the sparse autoencoder (SAE)
ae = trainSAE(
    data=buffer,
    trainer_configs=[
        {
            'trainer': StandardTrainer,
            'dict_class': AutoEncoder,
            'activation_dim': activation_dim,
            'dict_size': dictionary_size,
            'lr': 3e-4,
            'l1_penalty': 1e-3,
            'device': 'cuda:0',
            'layer': 1,
            'lm_name': 'EleutherAI/pythia-70m-deduped',
            'submodule_name': 'mlp'
        }
    ],
    save_dir='results'
)
