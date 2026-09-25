"""Small differentiable model for interface tests, independent of MLIP fitting."""
import os
import torch
from torch import nn
from ase import Atoms
from cace.models.atomistic import NeuralNetworkPotential
from cace.modules import Forces


class QuadraticEnergy(nn.Module):
    cutoff = 3.0
    model_outputs = ['energy', 'charges', 'bec']
    required_derivatives = []

    def forward(self, data):
        positions = data['positions']
        counts = len(data['ptr']) - 1
        energy = positions.new_zeros(counts)
        energy.index_add_(0, data['batch'], positions.square().sum(-1) / 2)
        data['energy'] = energy
        data['charges'] = positions[:, :1] * 0 + 0.25
        data['bec'] = positions * 0 + 1
        return data


def model():
    return NeuralNetworkPotential(representation=QuadraticEnergy(), output_modules=[Forces()])


def atoms():
    return Atoms('H2', positions=[[.2, .3, .4], [1.1, .7, .9]], cell=[6, 7, 8], pbc=True)


DEVICE = os.environ.get('CACE_TEST_DEVICE', 'cpu')


def compat_model(long_range=False, device=DEVICE):
    import cace
    from cace.representations import Cace
    representation = Cace(
        zs=[1, 5, 6, 7, 9], n_atom_basis=4, embed_receiver_nodes=True,
        atom_embedding_random_seed=[1234567, 1234567], cutoff=6.0,
        cutoff_fn=cace.modules.PolynomialCutoff(cutoff=6.0, p=6),
        radial_basis=cace.modules.BesselRBF(cutoff=6.0, n_rbf=6, trainable=True),
        n_radial_basis=12, max_l=3, max_nu=3, num_message_passing=1,
        type_message_passing=['Bchi'],
        args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
        avg_num_neighbors=10, device=torch.device(device))
    outputs = [cace.modules.Atomwise(n_layers=3, n_hidden=[32,16],
                                    output_key='sr_energy', descriptor_output_key='descriptor',
                                    use_batchnorm=False, add_linear_nn=True)]
    if long_range:
        outputs += [cace.modules.LesWrapper(energy_key='lr_energy'),
                    cace.modules.FeatureAdd(['sr_energy','lr_energy'], 'energy')]
    else:
        outputs += [cace.modules.FeatureAdd(['sr_energy'], 'energy')]
    outputs.append(Forces())
    return NeuralNetworkPotential(representation=representation, output_modules=outputs).to(device)


def synthetic_atoms():
    return Atoms(numbers=[1,5,6,7,9],
                 positions=[[.3,.4,.5],[1.4,.6,.8],[2.2,1.4,.4],[.8,2.1,1.7],[1.5,1.2,2.4]],
                 cell=[8.3,9.1,10.7], pbc=True)
