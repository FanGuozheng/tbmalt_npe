"""Interface to some popular ML framework."""
import torch
import pickle
from sklearn import linear_model, svm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tbmalt.common.batch import pack


class SciKitLearn:
    """Machine learning with optimized data.

    process data.
    perform ML prediction.
    """

    def __init__(self, system_train, system_pred, x_train, y_train, x_test,
                 bpnn=False, form='geometry', **kwargs):
        """Initialize."""
        assert x_train.dim(
        ) == 2, f'dim of x_train should be 2, get {x_train.dim()}'
        assert x_test.dim(
        ) == 2, f'dim of x_test should be 2, get {x_test.dim()}'
        assert y_train.dim(
        ) == 2, f'dim of y_train should be 2, get {y_train.dim()}'
        self.bpnn = bpnn
        self.form = form
        if self.bpnn:
            self.form = 'atom'
        self.save_model = kwargs.get('save_model', False)

        self.ml_method = kwargs.get('ml_method', 'linear')

        if not self.bpnn:
            size_sys = system_pred.n_atoms
            size_sys_train = system_train.n_atoms
            self.x_train = x_train
            self.y_train = y_train
            self.sum_size_test = [sum(size_sys[: ii])
                                  for ii in range(len(size_sys) + 1)]
            self.sum_size_train = [sum(size_sys_train[: ii])
                                   for ii in range(len(size_sys_train) + 1)]
            self.x_test = x_test
            self.prediction = getattr(SciKitLearn, self.ml_method)(self)
            self.model = self.reg
        else:
            self.uan = system_train.unique_atomic_numbers()
            assert (self.uan == system_pred.unique_atomic_numbers()).all()
            self.train_numbers = system_train.atomic_numbers[system_train.atomic_numbers.ne(
                0)]
            self.test_numbers = system_pred.atomic_numbers[system_pred.atomic_numbers.ne(
                0)]
            self.prediction, self.model = [], []
            for iuan in self.uan:
                self.x_test = x_test[self.test_numbers == iuan]
                self.x_train = x_train[self.train_numbers == iuan]
                self.y_train = y_train[self.train_numbers == iuan]
                ipred = getattr(SciKitLearn, self.ml_method)(self)
                self.prediction.append(ipred)
                self.model.append(self.reg)

        if self.save_model:
            filename = kwargs.get('model_name', 'model.pickle')
            pickle.dump(self.model, open(filename, 'wb'))

    def predict(self, xnew, geometry, imodel: int = None):
        if not self.bpnn:
            ynew = torch.from_numpy(self.model.predict(xnew))
        else:
            ynew = torch.from_numpy(self.model[imodel].predict(xnew))
        if self.form == 'geometry':
            size_sys_train = geometry.n_atoms
            sum_size_train = [sum(size_sys_train[: ii])
                              for ii in range(len(size_sys_train) + 1)]
            return pack([ynew[isize: sum_size_train[ii + 1]] for ii, isize in
                         enumerate(sum_size_train[: -1])])
        else:
            return ynew

    def linear(self):
        """Use the optimization dataset for training.

        Returns:
            linear ML method predicted DFTB parameters
        shape[0] of feature_data is defined by the optimized compression R
        shape[0] of feature_test is the defined by para['n_test']

        """
        self.reg = linear_model.LinearRegression()
        self.reg.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(self.reg.predict(self.x_test))

        if self.form == 'geometry':
            return pack([y_pred[isize: self.sum_size_test[ii + 1]]
                         for ii, isize in enumerate(self.sum_size_test[: -1])])
        else:
            return y_pred

    def svm(self):
        """ML process with support vector machine method."""
        self.reg = svm.SVR()
        self.reg.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(self.reg.predict(self.x_test))

        if self.form == 'geometry':
            return pack([y_pred[isize: self.sum_size_test[ii + 1]]
                         for ii, isize in enumerate(self.sum_size_test[: -1])])
        else:
            return y_pred

    def random_forest(self):
        """ML process with support vector machine method."""
        self.reg = RandomForestRegressor(n_estimators=100)
        self.reg.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(self.reg.predict(self.x_test))

        if self.form == 'geometry':
            return pack([y_pred[isize: self.sum_size_test[ii + 1]]
                         for ii, isize in enumerate(self.sum_size_test[: -1])])
        else:
            return y_pred

    def grad_boost(self):
        """ML process with support vector machine method."""
        self.reg = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1)
        size_sys = self.system.n_atoms
        self.reg.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(self.reg.predict(self.feature))

        if self.form == 'geometry':
            sum_size = [sum(size_sys[: ii])
                        for ii in range(len(size_sys) + 1)]
            return pack([y_pred[isize: sum_size[ii + 1]] for ii, isize in
                         enumerate(sum_size[: -1])]), self.reg
        else:
            return y_pred

    def krr(self):
        """Kernel ridge regression (KRR)."""
        self.reg = KernelRidge(alpha=1.0)
        self.reg.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(self.reg.predict(self.x_test))

        if self.form == 'geometry':
            return pack([y_pred[isize: self.sum_size_test[ii + 1]]
                         for ii, isize in enumerate(self.sum_size_test[: -1])])
        else:
            return y_pred

    def nn(self):
        """ML process with support vector machine method."""
        self.reg = MLPRegressor(solver='lbfgs', alpha=1e-5, max_iter=1000,
                                hidden_layer_sizes=(200, 2), random_state=1)
        self.reg.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(self.reg.predict(self.x_test))

        if self.form == 'geometry':
            return pack([y_pred[isize: self.sum_size_test[ii + 1]]
                         for ii, isize in enumerate(self.sum_size_test[: -1])])
        else:
            return y_pred

    def _flatten_target(self, target, size):
        """"""
        return torch.cat([itarget[: isize] for itarget, isize in
                          zip(target, size)])
