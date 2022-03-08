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
from tbmalt import Dftb2
from tbmalt.physics.dftb.slaterkoster import add_kpoint, hs_matrix_nn
from tbmalt.common.batch import pack
import tbmalt.common.maths as maths


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
                nn.Linear(self.n_feature, 500),
                self.activation,
                nn.Linear(500, 500),
                self.activation,
                nn.Linear(500, 500),
                self.activation,
                nn.Linear(500, 500),
                self.activation,
                nn.Linear(500, self.out_size),
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
        self.train_h = kwargs.get('train_h', True)
        self.train_s = kwargs.get('train_s', False)
        self.train_onsite = kwargs.get('train_onsite', False)

        self.l_pairs = torch.tensor([  # -> l1, l2
            [i, j] for i in range(int(np.sqrt(self.n_orbs)))
            for j in range(int(np.sqrt(self.n_orbs))) if i <= j])
        self.lm_pairs = torch.tensor([
            [i, j, k]  # -> l1, l2, m12
            for i in range(int(np.sqrt(self.n_orbs)))
            for j in range(int(np.sqrt(self.n_orbs)))
            for k in range(int(np.sqrt(self.n_orbs))) if i <= j and k <= i])

        # Build NN model list for different orbitals
        if self.train_h:
            self.h_models = self._build_orb_model()

        if self.train_onsite:
            self.onsite_feature = kwargs.get('onsite_feature', None)
            assert self.onsite_feature is not None, 'train onsite withou features'
            self.h_onsite_models = self._build_onsite_model()

        if self.train_s:
            self.s_models = self._build_orb_model()

        self.loss_fn = nn.L1Loss()
        self.pbc = pbc
        self.neig_resolve = kwargs.get('neig_resolve', True)
        self.is_pre_train = False

    def _build_orb_model(self):
        models = self.ss_model()
        if torch.max(self.l_pairs) >= 1:
            models.extend(self.sp_model())
            models.extend(self.pp_model())
        if torch.max(self.l_pairs) >= 2:
            models.extend(self.sd_model())
            models.extend(self.pd_model())
            models.extend(self.dd_model())
        if torch.max(self.l_pairs) >= 3:
            raise ValueError('do not support f orbitals, '
                             f'but get l number: {torch.max(self.l_pairs)}')
        return models

    def _build_onsite_model(self):
        models = self.s_onsite()
        if torch.max(self.l_pairs) >= 1:
            models.extend(self.p_onsite())
        if torch.max(self.l_pairs) >= 2:
            models.extend(self.d_onsite())
        return models

    def to_ham(self, hs_dict, type='H', train_onsite=False, hs_onsite={}, dftb=None):
        """Transfer predicted Hamiltonian to DFTB Hamiltonian."""
        dftb = dftb if dftb is not None else self.dftb
        size_dist = dftb.periodic.distances.shape
        hs_feed = dftb.h_feed if type == 'H' else dftb.s_feed
        for key, val in hs_dict.items():
            hs_dict[key] = val.reshape(*size_dist, val.shape[-1])
        return add_kpoint(hs_dict,
                          dftb.periodic,
                          dftb.basis,
                          hs_feed,
                          train_onsite=train_onsite,
                          hs_onsite=hs_onsite)

    def pre_train(self, X, isperiodic=True, device=torch.device('cpu')):
        # self.model.train()
        self.X = X
        self.is_pre_train = True
        _loss = []
        optimizer = self.set_optim(lr=8E-1)
        _mask = self.X[..., 0].gt(0.1)

        self.h_mat_dict, self.h_onsite_dict = \
            hs_matrix_nn(self.dftb.periodic,
                         self.dftb.basis,
                         self.dftb.h_feed,
                         train_onsite=self.train_onsite)
        if self.train_s:
            s_mat_dict, _ = hs_matrix_nn(self.dftb.periodic,
                                         self.dftb.basis,
                                         self.dftb.s_feed)
        distances = self.dftb.periodic.distances if isperiodic else \
            self.dftb.geometry.distances
        # rcut = 10
        # fc = 0.5 * (torch.cos(np.pi * distances[_mask_a].flatten()
        #                       / rcut) + 1.0).unsqueeze(-1)
        # fc[distances[_mask_a].flatten() > rcut] = 0
        fc = torch.exp(2-distances.flatten() * 0.5).unsqueeze(-1)

        # Compute prediction error
        for ii in range(500):
            if ii == 60:
                _mask = self.X[..., 0].gt(0.05)
                optimizer = self.set_optim(lr=5E-1)

            pred_h, pred_s, pred_h_on = {}, {}, {}
            for il, lm_pair in enumerate(self.lm_pairs.tolist()):
                # the 3rd parameter in the lm_pair equals to min(l), in this
                # case, if min(l) >= 1, like pp0, pp1, here will be gathered
                min_lm = int(min(lm_pair[:2]))
                key = tuple(lm_pair[:2])

                if lm_pair[-1] == 0:
                    pred_h.update(
                        {key: torch.zeros(self.X.shape[0], min_lm + 1)})
                    if self.train_s:
                        pred_s.update(
                            {key: torch.zeros(self.X.shape[0], min_lm + 1)})

                tmp = torch.zeros(self.X.shape[0], 1)
                tmp[_mask] = self.h_models[il](self.X[_mask])
                pred_h[key][..., lm_pair[-1]] = (tmp * fc).squeeze(-1)

                if self.train_s:
                    tmps = torch.zeros(self.X.shape[0], 1)
                    tmps[_mask] = self.s_models[il](self.X[_mask])
                    pred_s[key][..., lm_pair[-1]] = (tmps * fc).squeeze(-1)

            if self.train_onsite:
                for il in range(torch.max(self.lm_pairs).tolist() + 1):
                    pred_h_on.update(
                        {il: self.h_onsite_models[il](self.onsite_feature)})

            loss = 0
            for (predk, predv), (refk, refv) in zip(
                    pred_h.items(), self.h_mat_dict.items()):
                loss = loss + self.loss_fn(predv, refv.flatten(0, -2))

            if self.train_onsite and self.train_h:
                for il in range(torch.max(self.lm_pairs).tolist() + 1):
                    loss = loss + 0.005 * \
                        self.loss_fn(pred_h_on[il], self.h_onsite_dict[il])
                    if ii % 30 == 0:
                        print(il, pred_h_on[il])
                        print(self.h_onsite_dict[il])

            if self.train_s:
                for (predk, predv), (refk, refv) in zip(
                        pred_s.items(), s_mat_dict.items()):
                    loss = loss + 5 * self.loss_fn(predv, refv.flatten(0, -2))

            _loss.append(loss.detach())
            print('step: ', ii, 'loss: ', loss.detach().tolist())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ii % 30 == 0:
                for ii, val in self.h_mat_dict.items():
                    plt.plot(val.flatten(0, -2),
                             pred_h[ii].detach(), '.', label='H'+str(ii))
                    plt.plot([-0.2, 0.2], [-0.2, 0.2], 'k')
                    _h_data = val.flatten()
                    print('key: ', ii, 'len:', len(_h_data), 'error H:', abs(
                        _h_data - pred_h[ii].detach().flatten()).sum())
                plt.xlabel('DFTB ham')
                plt.ylabel('predicted ham')
                plt.legend()
                plt.show()
                if self.train_s:
                    for ii in s_mat_dict.keys():
                        plt.plot(s_mat_dict[ii].flatten(0, -2),
                                 pred_s[ii].detach(), '.', label='S' + str(ii))
                        plt.plot([-0.5, 0.4], [-0.5, 0.4], 'k')
                        _s_data = s_mat_dict[ii].flatten()
                        print('key: ', ii, 'len:', len(_s_data), 'error S:',
                              abs(_s_data - pred_s[ii].detach().flatten()).sum())
                    plt.xlabel('DFTB s')
                    plt.ylabel('predicted s')
                    plt.legend()
                    plt.show()
                    for ii in s_mat_dict.keys():
                        plt.plot(X[..., 0], pred_s[ii].detach(),
                                 '.', label='S' + str(ii))
                    plt.xlabel('X0')
                    plt.ylabel('predicted S')
                    plt.legend()
                    plt.show()

        plt.plot(torch.arange(len(_loss)), _loss, label='loss')
        plt.legend()
        plt.show()

    def forward(self, X, ref, isperiodic,
                n_band0, n_band1,
                n_valence=3, n_conduction=1, klines=None, path_to_skf=None):
        dftb_band = self.dftb.eigenvalue.clone()
        dftb_h = self.dftb.ham.flatten().real
        dftb_s = self.dftb.over.flatten().real

        # Select standard DFTB bands
        dftb_band = dftb_band - self.dftb.E_fermi.unsqueeze(-1).unsqueeze(-1)
        ind0 = torch.repeat_interleave(
            torch.arange(len(dftb_band)), n_valence + n_conduction)
        n_val = dftb_band.sum(-2).lt(0).sum(-1)
        ind_1 = (torch.repeat_interleave(n_val, n_valence + n_conduction) +
                 torch.arange(-n_valence, n_conduction).repeat(len(dftb_band))).flatten()
        dftb_band = pack(dftb_band[ind0, :, ind_1].split(
            tuple((torch.ones(len(dftb_band)) * (n_valence + n_conduction)).int()), 0))

        optimizer = self.set_optim(lr=1E-4)
        self.ref = ref
        _loss = []

        distances = self.dftb.periodic.distances if isperiodic else \
            self.dftb.geometry.distances
        # rcut = 10
        # fc = 0.5 * (torch.cos(np.pi * distances.flatten() / rcut) + 1.0).unsqueeze(-1)
        # fc[distances.flatten() > rcut] = 0
        fc = torch.exp(2-distances.flatten() * 0.5).unsqueeze(-1)

        # Compute prediction error
        _mask = X[..., 0].gt(0.05)
        error_hs = {(0, 0): 0.02, (0, 1): 0.03, (0, 2): 0.04,
                    (1, 1): 0.04, (1, 2): 0.06, (2, 2): 0.1}
        for ii in range(100):
            # if ii == 40:
            #     optimizer = self.set_optim(lr=1E-5)

            pred_h, pred_s, pred_h_on = {}, {}, {}
            for il, lm_pair in enumerate(self.lm_pairs.tolist()):
                # the 3rd parameter in the lm_pair equals to min(l), in this
                # case, if min(l) >= 1, like pp0, pp1, here will be gathered
                min_lm = int(min(lm_pair[:2]))
                key = tuple(lm_pair[:2])
                if lm_pair[-1] == 0:
                    pred_h.update({key: torch.zeros(X.shape[0], min_lm + 1)})
                    if self.train_s:
                        pred_s.update(
                            {key: torch.zeros(X.shape[0], min_lm + 1)})

                tmp = torch.zeros(X.shape[0], 1)
                tmp[_mask] = self.h_models[il](X[_mask])

                pred_h[key][..., lm_pair[-1]] = (tmp * fc).squeeze(-1)
                if self.train_s:
                    tmps = torch.zeros(X.shape[0], 1)
                    tmps[_mask] = self.s_models[il](X[_mask])

                    # NN will give negtive value
                    pred_s[key][..., lm_pair[-1]] = (tmps * fc).squeeze(-1)

                # Make constrains for each orbital based on DFTB H & S
                if min(key) == lm_pair[-1]:
                    pred_h[key] = torch.clamp(
                        pred_h[key],
                        self.h_mat_dict[key].flatten(
                            0, -2) - fc * error_hs[key],
                        self.h_mat_dict[key].flatten(0, -2) + fc * error_hs[key])

            if self.train_onsite:
                error_on = {0: 0.005, 1: 0.008, 2: 0.01}
                for il in range(torch.max(self.lm_pairs).tolist() + 1):
                    ih_on = self.h_onsite_models[il](self.onsite_feature)
                    if ii % 30 == 0:
                        print('il', il, ih_on)
                    ih_on = torch.clamp(ih_on,
                                        self.h_onsite_dict[il] - error_on[il],
                                        self.h_onsite_dict[il] + error_on[il])
                    pred_h_on.update({il: ih_on})

            # < k_i | H_ij | k_j >
            ham = self.to_ham(pred_h, 'H', self.train_onsite, pred_h_on)
            dftb = Dftb2(self.dftb.geometry,
                         self.dftb.shell_dict,
                         path_to_skf=path_to_skf,
                         skf_type='skf',
                         klines=klines,
                         maxiter=1)
            if self.train_s:
                over = self.to_ham(pred_s, 'S')
                # while True:
                #     if iadd > 0:
                over = over + torch.eye(over.shape[-2]).repeat(
                    over.shape[0], over.shape[-1], 1, 1).permute(0, 2, 3, 1) * 0.05

                self.dftb(hamiltonian=ham, overlap=over)
            else:
                self.dftb(hamiltonian=ham)

            pred_band0 = self.dftb.eigenvalue
            pred_band0 = pred_band0 - \
                self.dftb.E_fermi.unsqueeze(-1).unsqueeze(-1).detach()
            ind0 = torch.repeat_interleave(
                torch.arange(len(pred_band0.detach())), n_valence + n_conduction)
            n_val = pred_band0.detach().sum(-2).lt(0).sum(-1)
            ind_1 = (torch.repeat_interleave(n_val, n_valence + n_conduction) +
                     torch.arange(-n_valence, n_conduction).repeat(
                         len(pred_band0.detach()))).flatten()
            pred_band0 = pack(pred_band0[ind0, :, ind_1].split(tuple((
                torch.ones(len(pred_band0.detach())) * (n_valence + n_conduction)).int()), 0))
            pred_band1 = pred_band0[..., n_band0]
            delta_band = pred_band0[..., n_band1] - pred_band1

            loss = 0
            loss = self.loss_fn(pred_band1, ref['band'])
            loss = loss + 0.5 * self.loss_fn(delta_band, ref['delta_band'])

            # pred_band = pred_band.clone() + shift.unsqueeze(-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _loss.append(loss.detach().tolist())
            print('step: ', ii, 'loss: ', loss.detach().tolist())

            if ii % 30 == 0:
                # print('ham', ham[..., 0].real)
                # print('dftb.ham', self.dftb.ham[..., 0].real)
                for ir, idftb, ip in zip(ref['tot_band'],
                                         dftb_band.numpy(),
                                         pred_band0.detach().numpy()):
                    plt.plot(torch.arange(len(ir.T)), ir.T, color='r')
                    plt.plot(torch.arange(len(idftb.T)),
                             idftb.T, color='g', linestyle='--')
                    plt.plot(torch.arange(len(ip.T)), ip.T,
                             color='c', linestyle='-.')
                    plt.plot([0], [-10], color='r', label='ref')
                    plt.plot([0], [-10], color='g', label='dftb')
                    plt.plot([0], [-10], color='c', label='pred')
                    # plt.plot([0, 120], [0, 0], color='k')
                    plt.ylim(-5, 2)
                    plt.xlim(0, 90)
                    plt.legend()
                    plt.show()

                plt.plot(dftb_h.numpy(), ham.detach().flatten().numpy().real,
                         'b.', label='ham ' + str(ii))
                plt.plot([-0.5, 0.4], [-0.5, 0.4], 'k')
                plt.xlabel('DFTB ham')
                plt.ylabel('predicted ham')
                plt.legend()
                plt.show()
                if self.train_s:
                    plt.plot(dftb_s, over.detach().flatten().real,
                             'b.', label='over ' + str(ii))
                    plt.plot([-0.8, 0.8], [-0.8, 0.8], 'k')
                    plt.xlabel('DFTB S')
                    plt.ylabel('predicted S')
                    plt.legend()
                    plt.show()

                for key, val in pred_h.items():
                    print('avg pred', val.sum() / torch.numel(val))
                    plt.plot(X[..., 0], val.detach().flatten(0, -2), '.')
                plt.xlabel('ACSF-G0')
                plt.ylabel('predicted ham')
                plt.legend()
                plt.show()

        plt.plot(torch.arange(len(_loss)), torch.log(torch.tensor(_loss)))
        plt.show()

    def test(self, X, dftb_test, ref, onsite_feature=None, isperiodic=True,
             n_band0=None, n_band1=None,
             n_valence=3, n_conduction=1):

        _mask = X[..., 0].gt(0.05)
        distances = dftb_test.periodic.distances if isperiodic else \
            dftb_test.geometry.distances
        fc = torch.exp(2-distances.flatten() * 0.5).unsqueeze(-1)
        pred_h, pred_s, pred_h_on = {}, {}, {}
        for il, lm_pair in enumerate(self.lm_pairs.tolist()):
            # the 3rd parameter in the lm_pair equals to min(l), in this
            # case, if min(l) >= 1, like pp0, pp1, here will be gathered
            min_lm = int(min(lm_pair[:2]))
            key = tuple(lm_pair[:2])
            if lm_pair[-1] == 0:
                pred_h.update({key: torch.zeros(X.shape[0], min_lm + 1)})
                if self.train_s:
                    pred_s.update({key: torch.zeros(X.shape[0], min_lm + 1)})

            tmp = torch.zeros(X.shape[0], 1)
            tmp[_mask] = self.h_models[il](X[_mask])
            pred_h[key][..., lm_pair[-1]] = (tmp * fc).squeeze(-1)
            if self.train_s:
                tmps = torch.zeros(X.shape[0], 1)
                tmps[_mask] = self.s_models[il](X[_mask])

                # NN will give negtive value
                pred_s[key][..., lm_pair[-1]] = (tmps * fc).squeeze(-1)

        if self.train_onsite:
            for il in range(torch.max(self.lm_pairs).tolist() + 1):
                pred_h_on.update({il: self.h_onsite_models[il](onsite_feature)})

        # < k_i | H_ij | k_j >
        ham = self.to_ham(pred_h, 'H', self.train_onsite, pred_h_on, dftb_test)
        if self.train_s:
            over = self.to_ham(pred_s, 'S')
            over = over + torch.eye(over.shape[-2]).repeat(
                over.shape[0], over.shape[-1], 1, 1).permute(0, 2, 3, 1) * 0.05

            dftb_test(hamiltonian=ham, overlap=over)
        else:
            dftb_test(hamiltonian=ham)

        # pred_band0 = self.dftb.eigenvalue.permute(1, 2, 0)[..., 0]
        pred_band0 = dftb_test.eigenvalue
        print('dftb_test.fermi', dftb_test.E_fermi, pred_band0.shape)
        pred_band0 = pred_band0 - dftb_test.E_fermi.unsqueeze(-1).unsqueeze(-1)
        ind0 = torch.repeat_interleave(
            torch.arange(len(pred_band0.detach())), n_valence + n_conduction)
        n_val = pred_band0.detach().sum(-2).lt(0).sum(-1)
        ind_1 = (torch.repeat_interleave(n_val, n_valence + n_conduction) +
                 torch.arange(-n_valence, n_conduction).repeat(
                     len(pred_band0.detach()))).flatten()
        pred_band0 = pack(pred_band0[ind0, :, ind_1].split(
            tuple((torch.ones(len(pred_band0.detach())) * (
                n_valence + n_conduction)).int()), 0))
        pred_band1 = pred_band0[..., n_band0]
        delta_band = pred_band0[..., n_band1] - pred_band1

        # Select standard DFTB bands
        dftb_test()
        dftb_band = dftb_test.eigenvalue.clone()
        dftb_band = dftb_band - dftb_test.E_fermi.unsqueeze(-1).unsqueeze(-1)
        ind0 = torch.repeat_interleave(
            torch.arange(len(dftb_band)), n_valence + n_conduction)
        n_val = dftb_band.sum(-2).lt(0).sum(-1)
        ind_1 = (torch.repeat_interleave(n_val, n_valence + n_conduction) +
                 torch.arange(-n_valence, n_conduction).repeat(len(dftb_band))).flatten()
        dftb_band = pack(dftb_band[ind0, :, ind_1].split(
            tuple((torch.ones(len(dftb_band)) * (n_valence + n_conduction)).int()), 0))

        for ir, idftb, ip in zip(ref['tot_band'],
                                 dftb_band.detach().numpy(),
                                 pred_band0.detach().numpy()):
            plt.plot(torch.arange(len(ir.T)), ir.T, color='r')
            plt.plot(torch.arange(len(idftb.T)),
                     idftb.T, color='g', linestyle='--')
            plt.plot(torch.arange(len(ip.T)), ip.T, color='c', linestyle='--')
            plt.plot([0], [-10], color='r', label='ref')
            plt.plot([0], [-10], color='g', label='dftb')
            plt.plot([0], [-10], color='c', label='pred')
            # plt.plot([0, 120], [0, 0], color='k')
            plt.ylim(-5, 2)
            plt.xlim(0, 90)
            plt.legend()
            plt.show()

    def set_optim(self, lr=1E-5):
        train_params = []
        if self.train_h:
            train_params = train_params + list(self.h_models.parameters())

        if self.train_s:
            train_params = train_params + list(self.s_models.parameters())

        if self.train_onsite:
            train_params = train_params + list(self.h_onsite_models.parameters())

        return torch.optim.SGD(train_params, lr=lr)

    def s_onsite(self):
        """Return ss0 model."""
        return nn.ModuleList([MLP(n_feature=self.n_feature,
                                  out_size=1,
                                  activation=self.activation,
                                  nn_type=self.nn_type)])

    def p_onsite(self):
        """Return ss0 model."""
        return nn.ModuleList([MLP(n_feature=self.n_feature,
                                  out_size=1,
                                  activation=self.activation,
                                  nn_type=self.nn_type)])

    def d_onsite(self):
        """Return ss0 model."""
        return nn.ModuleList([MLP(n_feature=self.n_feature,
                                  out_size=1,
                                  activation=self.activation,
                                  nn_type=self.nn_type)])

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


# if __name__ == '__main__':
#     test = Test(3, 1)
#     test(torch.randn(10, 3), ref=torch.randn(10, 1))
