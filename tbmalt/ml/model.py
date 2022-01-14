#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:49:07 2021

@author: gz_fan
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tbmalt.common.maths import hellinger
from tbmalt.physics.dftb.slaterkoster import add_kpoint, hs_matrix_nn
from tbmalt.common.batch import pack

# Define model
class MLP(nn.Module):

    def __init__(self, n_feature: int = 4, out_size: int = 1,
                 activation='ReLU', nn_type='linear'):
        super(MLP, self).__init__()
        self.n_feature = n_feature
        self.flatten = nn.Flatten()
        self.activation = getattr(nn, activation)()
        self.out_size = out_size
        self._nn = self.nn_type(nn_type)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.flatten(x)
        # x = self.dropout(x)
        pred = self._nn(x)
        return pred

    def nn_type(self, type='linear'):
        if type == 'linear':
            torch.manual_seed(1)
            return nn.Sequential(
                nn.Linear(self.n_feature, 1000),
                self.activation,
                nn.Linear(1000, 1000),
                self.activation,
                nn.Linear(1000, 1000),
                self.activation,
                nn.Linear(1000, 1000),
                self.activation,
                nn.Linear(1000, self.out_size),
            )
        elif type == 'cnn':
            return nn.Sequential(
                nn.Conv1d(self.n_feature, self.out_size, 100),
                nn.BatchNorm2d(self.n_feature),
                self.activation,
            )


class Hamiltonian(nn.Module):

    def __init__(self, n_feature: int = 4, out_size: int = 1, activation='ReLU',
                 nn_type='linear', pbc=True, dftb=None, **kwargs):
        super(Hamiltonian, self).__init__()
        self.n_feature = n_feature
        self.out_size = out_size
        self.activation = activation
        self.nn_type = nn_type
        self.dftb = dftb
        self.n_orbs = torch.max(self.dftb.basis.orbs_per_atom).tolist()

        self.l_pairs = torch.tensor([  # -> l1, l2
            [i, j] for i in range(int(np.sqrt(self.n_orbs)))
            for j in range(int(np.sqrt(self.n_orbs))) if i <= j])
        self.lm_pairs = torch.tensor([
            [i, j, k]  # -> l1, l2, m12
            for i in range(int(np.sqrt(self.n_orbs)))
            for j in range(int(np.sqrt(self.n_orbs)))
            for k in range(int(np.sqrt(self.n_orbs))) if i <= j and k <= i])

        # Build NN model list for different orbitals
        self.models = self.ss_model()
        if torch.max(self.l_pairs) >= 1:
            self.models.extend(self.sp_model())
            self.models.extend(self.pp_model())
        if torch.max(self.l_pairs) >= 2:
            self.models.extend(self.sd_model())
            self.models.extend(self.pd_model())
            self.models.extend(self.dd_model())
        if torch.max(self.l_pairs) >= 3:
            raise ValueError('do not support f orbitals, '
                             f'but get l number: {torch.max(self.l_pairs)}')

        self.loss_fn = nn.L1Loss()
        self.pbc = pbc
        self.neig_resolve = kwargs.get('neig_resolve', True)
        self.is_pre_train = False

    def to_ham(self, hs_dict):
        """Transfer predicted Hamiltonian to DFTB Hamiltonian."""
        size_dist = self.dftb.periodic.distances.shape

        for key, val in hs_dict.items():
            hs_dict[key] = val.reshape(*size_dist, val.shape[-1])
        return add_kpoint(hs_dict,
                          self.dftb.periodic,
                          self.dftb.basis,
                          self.dftb.h_feed)

    def pre_train(self, X, isperiodic=True, device=torch.device('cpu')):
        # self.model.train()
        self.X = X
        self.is_pre_train = True
        _loss = []
        optimizer = torch.optim.SGD(self.models.parameters(), lr=8e-1)
        _mask = self.X[..., 0].gt(0.1)

        if isperiodic:
            mat_dict = hs_matrix_nn(self.dftb.periodic,
                                    self.dftb.basis,
                                    self.dftb.h_feed)
            distances = self.dftb.periodic.distances
            # rcut = 10
            # fc = 0.5 * (torch.cos(np.pi * distances[_mask_a].flatten()
            #                       / rcut) + 1.0).unsqueeze(-1)
            # fc[distances[_mask_a].flatten() > rcut] = 0
            fc = torch.exp(2-distances.flatten() * 0.5).unsqueeze(-1)

        # Compute prediction error
        for ii in range(120):
            if ii == 40:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=5e-1)
                _mask = self.X[..., 0].gt(0.05)

            pred = {}
            for lm_pair, model in zip(self.lm_pairs.tolist(), self.models):
                # the 3rd parameter in the lm_pair equals to min(l), in this
                # case, if min(l) >= 1, like pp0, pp1, here will be gathered
                min_lm = int(min(lm_pair[:2]))
                key = tuple(lm_pair[:2])

                if lm_pair[-1] == 0:
                    pred.update({key: torch.zeros(self.X.shape[0], min_lm + 1)})

                tmp = torch.zeros(self.X.shape[0], 1)
                tmp[_mask] = model(self.X[_mask])
                pred[key][..., lm_pair[-1]] = (tmp * fc).squeeze(-1)

            loss = 0
            for (predk, predv), (refk, refv) in zip(pred.items(), mat_dict.items()):
                loss = loss + self.loss_fn(predv, refv.flatten(0, -2))

            _loss.append(loss.detach())
            print('loss:', ii, loss.detach())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ii % 30 == 0:
                for ii in mat_dict.keys():
                    plt.plot(mat_dict[ii].flatten(0, -2),
                             pred[ii].detach(), 'b.', label='ham ' + str(ii))
                    plt.plot([-0.2, 0.2], [-0.2, 0.2], 'k')
                plt.xlabel('DFTB ham')
                plt.ylabel('predicted ham')
                plt.legend()
                plt.show()
        plt.plot(torch.arange(len(_loss)), _loss, label='loss')
        plt.legend()
        plt.show()


    def forward(self, X, ref=None, isperiodic=True):
        optimizer = torch.optim.SGD(self.models.parameters(), lr=2e-5)
        # self.model.train()
        self.ref = ref
        _loss = []

        if isperiodic:
            distances = self.dftb.periodic.distances
            # rcut = 10
            # fc = 0.5 * (torch.cos(np.pi * distances.flatten() / rcut) + 1.0).unsqueeze(-1)
            # fc[distances.flatten() > rcut] = 0
            fc = torch.exp(2-distances.flatten() * 0.5).unsqueeze(-1)

        # Compute prediction error
        _mask = X[..., 0].gt(0.05)
        for ii in range(120):
            if ii == 30:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=1e-5)
            if ii == 60:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=5e-6)

            pred = {}
            for lm_pair, model in zip(self.lm_pairs.tolist(), self.models):
                # the 3rd parameter in the lm_pair equals to min(l), in this
                # case, if min(l) >= 1, like pp0, pp1, here will be gathered
                min_lm = int(min(lm_pair[:2]))
                key = tuple(lm_pair[:2])
                if lm_pair[-1] == 0:
                    pred.update({key: torch.zeros(X.shape[0], min_lm + 1)})

                tmp = torch.zeros(X.shape[0], 1)
                tmp[_mask] = model(X[_mask])
                pred[key][..., lm_pair[-1]] = (tmp * fc).squeeze(-1)

            # if self.neig_resolve:
            #     pred.clamp_(min=-0.5, max=0)
            # < k_i | H_ij | k_j >
            ham = self.to_ham(pred)
            self.dftb(hamiltonian=ham)
            # pred_band0 = self.dftb.eigenvalue.permute(1, 2, 0)[..., 0]

            pred_band0 = self.dftb.eigenvalue.permute(1, 0, 2)
            pred_band0[0] = pred_band0[0] - torch.max(pred_band0[0, :, 3]).detach()
            # pred_band0[1] = pred_band0[1] - torch.max(pred_band0[1, :, 11]).detach()  # 11th state
            pred_band1 = pack([pred_band0[0, [0, 10, 18]][..., : 7]])
                               # pred_band0[1, [0, 10]][..., 4: 15]])
            pred_band2 = pack([pred_band0[0, [1, 11, 19]][..., : 7]])
                               # pred_band0[1, [1, 11]][..., 4: 15]])
            delta_band = pred_band2 - pred_band1
            # # Consider the shift
            # shift = (pred_band0 - _ref).sum(-1).detach() / pred_band0.shape[-1]
            # pred_band = pred_band0 - shift.unsqueeze(-1)
            # loss = self.loss_fn(pred_band, _ref) + 0.25 * self.loss_fn(pred_band0, _ref)
            # loss = self.loss_fn(pred_band0, _ref)

            loss = self.loss_fn(pred_band1, ref['band'])
            loss = loss + 0.5 * self.loss_fn(delta_band, ref['delta_band'])

            # pred_band = pred_band.clone() + shift.unsqueeze(-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _loss.append(loss.detach())
            print('loss:', ii, loss.detach())

            if ii % 30 == 0:
                # for param in self.model.parameters():
                #     print('param', param.data.shape, torch.max(param.data),
                #           torch.min(param.data), 'grad', torch.max(param.grad))
                #     with torch.no_grad():
                #         param.clamp_(min=-0.5, max=0.5)

                self.dftb()
                dftb_band = self.dftb.eigenvalue.permute(1, 0, 2)
                dftb_band[0] = dftb_band[0] - torch.max(dftb_band[0, :, 3])
                # dftb_band[1] = dftb_band[1] - torch.max(dftb_band[1, :, 11])
                # plt.plot(ref[0].flatten(), pred_band0.detach().flatten(),
                #          'rx', label=str(ii))
                # plt.xlabel('reference band')
                # plt.ylabel('predicted band')
                # plt.legend()
                # plt.show()
                print('pred_band1', pred_band1, '\n ref[1]', ref['band'])
                print('dftb_band', dftb_band[0, [0, 10, 18]][..., :7])
                # print('ham', ham[..., 0].real)
                # print('dftb.ham', self.dftb.ham[..., 0].real)
                for ir, idftb in zip(ref['tot_band'], dftb_band):
                    plt.plot(torch.arange(len(ir)), ir, color='r')
                    plt.plot(torch.arange(len(idftb)), idftb, color='g', linestyle='--')
                    plt.plot([0], [-10], color='r', label='ref')
                    plt.plot([0], [-10], color='g', label='dftb')
                    plt.plot([0, 19], [0, 0], color='k')
                    plt.ylim(-15, 5)
                    plt.legend()
                    plt.show()
                for ir, ip in zip(ref['tot_band'], pred_band0.detach()):
                    plt.plot(torch.arange(len(ir)), ir, color='r')
                    plt.plot(torch.arange(len(ip)), ip, color='c', linestyle='--')
                    plt.plot([0], [-10], color='r', label='ref')
                    plt.plot([0], [-10], color='c', label='pred')
                    plt.plot([0, 19], [0, 0], color='k')
                    plt.ylim(-15, 5)
                    plt.legend()
                    plt.show()

                plt.plot(self.dftb.ham.flatten().real, ham.detach().flatten().real,
                         'b.', label='ham ' + str(ii))
                plt.plot([-0.5, 0.4], [-0.5, 0.4], 'k')
                plt.xlabel('DFTB ham')
                plt.ylabel('predicted ham')
                plt.legend()
                plt.show()

                for key, val in pred.items():
                    print('avg pred', val.sum() / torch.numel(val))
                    plt.plot(X[..., 0], val.detach().flatten(0, -2), '.')
                plt.xlabel('ACSF-G0')
                plt.ylabel('predicted ham')
                plt.legend()
                plt.show()

        plt.plot(torch.arange(len(_loss)), torch.log(torch.tensor(_loss)))
        plt.show()
        # for ib, ip, ih in zip(ref, pred_band1.detach(), ham.detach()):
        #     plt.plot(torch.arange(len(ib)), ib, color='r')
        #     plt.plot(torch.arange(len(ip)), ip.T, color='c')
        #     plt.show()

    def ss_model(self):
        """Return ss0 model."""
        return nn.ModuleList([MLP(n_feature=self.n_feature,
                                  out_size=self.out_size,
                                  activation=self.activation,
                                  nn_type=self.nn_type)])

    def sp_model(self):
        """Return sp0 model."""
        return nn.ModuleList([MLP(n_feature=self.n_feature,
                                  out_size=self.out_size,
                                  activation=self.activation,
                                  nn_type=self.nn_type)])

    def sd_model(self):
        """Return sd0 model."""
        return nn.ModuleList([MLP(n_feature=self.n_feature,
                                  out_size=self.out_size,
                                  activation=self.activation,
                                  nn_type=self.nn_type)])

    def pp_model(self):
        """Return pp0, pp1 models."""
        return nn.ModuleList([MLP(n_feature=self.n_feature,
                                  out_size=self.out_size,
                                  activation=self.activation,
                                  nn_type=self.nn_type),
                              MLP(n_feature=self.n_feature,
                                  out_size=self.out_size,
                                  activation=self.activation,
                                  nn_type=self.nn_type)])

    def pd_model(self):
        """Return pd0, pd1 models."""
        return nn.ModuleList([MLP(n_feature=self.n_feature,
                                  out_size=self.out_size,
                                  activation=self.activation,
                                  nn_type=self.nn_type),
                              MLP(n_feature=self.n_feature,
                                  out_size=self.out_size,
                                  activation=self.activation,
                                  nn_type=self.nn_type)])

    def dd_model(self):
        """Return dd0, dd1, dd2 models."""
        return nn.ModuleList([MLP(n_feature=self.n_feature,
                                  out_size=self.out_size,
                                  activation=self.activation,
                                  nn_type=self.nn_type),
                              MLP(n_feature=self.n_feature,
                                  out_size=self.out_size,
                                  activation=self.activation,
                                  nn_type=self.nn_type),
                              MLP(n_feature=self.n_feature,
                                  out_size=self.out_size,
                                  activation=self.activation,
                                  nn_type=self.nn_type)])


class Test(nn.Module):

    def __init__(self, n_feature: int = 4, out_size: int = 1, activation='ReLU',
                 nn_type='linear'):
        super(Test, self).__init__()
        self.n_feature = n_feature
        self.out_size = out_size
        self.n_orbs = torch.range(1, 5)

        # Build NN model list for different orbitals
        self.models = nn.ModuleList(
            [MLP(n_feature=self.n_feature,
                 out_size=self.out_size,
                 activation=activation,
                 nn_type=nn_type)] * len(self.n_orbs))
        self.loss_fn = nn.L1Loss()

    def forward(self, X, ref=None):
        optimizer = torch.optim.SGD(self.models.parameters(), lr=5e-5)
        # self.model.train()
        self.ref = ref
        _loss = []

        # Compute prediction error
        for ii in range(5):
            if ii <= 30:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=6e-5)
            elif ii <= 60:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=4e-5)
            elif ii <= 90:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=3e-5)
            elif ii <= 120:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=2e-5)

            pred = {}
            for lm_pair, model in zip(self.n_orbs.tolist(), self.models):
                # print('lm_pair', lm_pair)
                pred.update({int(lm_pair): model(X)})
            print('model', self.models[-1] is self.models[-2])

            for ii, key in enumerate(pred.keys()):
                if ii == 0:
                    eigen = pred[key]
                else:
                    eigen = eigen + pred[key] ** ii + pred[key] * ii
                print(ii, 'pred[key] ** ii', (pred[key]).sum().detach())
            # # Consider the shift
            loss = self.loss_fn(eigen, ref)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _loss.append(loss.detach())
            print('loss:', ii, loss.detach())
            print([(key, val.sum()) for key, val in pred.items()])



# if __name__ == '__main__':
#     test = Test(3, 1)
#     test(torch.randn(10, 3), ref=torch.randn(10, 1))
