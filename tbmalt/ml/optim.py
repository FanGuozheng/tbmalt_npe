"""Train code."""
from typing import Literal, Dict
import logging
import numpy as np
import torch
from torch import Tensor, nn
import matplotlib.pyplot as plt
from tbmalt import Geometry, Dftb1, Dftb2, SkfParamFeed, Shell
from tbmalt.common.maths import hellinger
from tbmalt.common.batch import pack
from tbmalt.ml.skfeeds import SkfFeed, VcrFeed, TvcrFeed
from tbmalt.physics.dftb.slaterkoster import hs_matrix
from tbmalt.ml.feature import Dscribe
from tbmalt.ml.scikitlearn import SciKitLearn
from tbmalt.structures.periodic import Periodic
from tbmalt.common.logger import get_logger


class Model(nn.Module):
    """Optimizer template for DFTB parameters training.

    This is a template class inherited from "nn.Module". The function
    "__init__" will define machine learning parameters including optimizer,
    loss function, etc.

    Arguments:
        geometry: TBMaLT geometry object.
        reference: A dictionary for reference data.
        variables: A list of variables with gradients.
        params: Dictionary which stores ML parameters.
        tolerance: Accuracy for machine learning loss convergence.

    """

    def __init__(
        self,
        geometry: Geometry,
        reference: dict,
        variables: list,
        params: dict,
        logger=None,
        **kwargs,
    ):
        super(Model, self).__init__()
        self.geometry = geometry
        self.batch_size = self.geometry._n_batch
        self.reference = reference

        self.variable = variables

        self.params = params
        self.tolerance = (
            self.params["tolerance"] if "tolerance" in self.params.keys() else 1e-6
        )
        self.logger = get_logger(__name__) if logger is None else logger

        # Initialize all targets with None
        for target in self.params["targets"]:
            setattr(self, target, None)

        self.lr = self.params["lr"]

        # get loss function
        self.criterion = getattr(torch.nn, self.params["loss_function"])(
            reduction="mean"
        )

        # get optimizer
        self.optimizer = getattr(torch.optim, self.params["optimizer"])(
            self.variable, lr=self.lr
        )

    def forward(self, **kwargs):
        """Call train class with properties."""
        self.logger.info("Start training...")
        self.loss_list = []
        self.loss_list.append(0)
        self.steps = self.params["max_steps"]

    def __loss__(self, results, scc=True):
        """Get loss function for single step."""
        self.loss = 0.0

        # add properties (ML targetss) to loss function
        for target in self.params["targets"]:
            if target == "band":
                pred_band0 = results.eigenvalue

                pred_band0 = (
                    pred_band0 - self.dftb.E_fermi.unsqueeze(-1).unsqueeze(-1).detach()
                )
                ind0 = torch.repeat_interleave(
                    torch.arange(len(pred_band0.detach())),
                    self.params["n_valence"] + self.params["n_conduction"],
                )
                n_val = pred_band0.detach().sum(-2).lt(0).sum(-1)
                ind_1 = (
                    torch.repeat_interleave(
                        n_val, self.params["n_valence"] + self.params["n_conduction"]
                    )
                    + torch.arange(
                        -self.params["n_valence"], self.params["n_conduction"]
                    ).repeat(len(pred_band0.detach()))
                ).flatten()
                pred_band0 = pack(
                    pred_band0[ind0, :, ind_1].split(
                        tuple(
                            (
                                torch.ones(len(pred_band0.detach()))
                                * (
                                    self.params["n_valence"]
                                    + self.params["n_conduction"]
                                )
                            ).int()
                        ),
                        0,
                    )
                )
                print("pred_band0", pred_band0.shape, self.params["n_band0"])
                pred_band1 = pred_band0[..., self.params["n_band0"]]
                delta_band = pred_band0[..., self.params["n_band1"]] - pred_band1
                self.loss = self.loss + self.criterion(
                    pred_band1, self.reference["band"]
                )
                self.loss = self.loss + 0.5 * self.criterion(
                    delta_band, self.reference["delta_band"]
                )
                print(
                    "pred_band1",
                    pred_band1,
                    "self.reference",
                    self.reference["band"],
                )
                for ir, ip in zip(
                    self.reference["tot_band"], pred_band0.detach().numpy()
                ):
                    plt.plot(torch.arange(len(ir.T)), ir.T, color="r")
                    plt.plot(torch.arange(len(ip.T)), ip.T, color="c", linestyle="-.")
                    plt.plot([0], [-10], color="r", label="ref")
                    plt.plot([0], [-10], color="c", label="pred")
                    plt.ylim(-5, 2)
                    plt.xlim(0, 90)
                    plt.legend()
                    plt.show()

            else:
                self.loss = (
                    self.loss
                    + self.criterion(
                        results.__getattribute__(target), self.reference[target]
                    )
                    * self.params[target + "_weight"]
                )
                setattr(self, target, results.__getattribute__(target).detach())

        self.loss_list.append(self.loss.detach())
        self.reach_convergence = (
            abs(self.loss_list[-1] - self.loss_list[-2]) < self.tolerance
        )

    def _predict_onsite(
        self, geometry_pred, x_train, x_test, basis_pred, ml_method="nn", **kwargs
    ) -> Tensor:
        """Predict with training results."""
        train_onsite_type = kwargs.get("train_onsite_type", "element")
        opt_onsite = self.h_feed.on_site_dict["ml_onsite"].detach().clone()
        mask = self.geometry.atomic_numbers.ne(0)
        opt_onsite = pack(
            torch.split(opt_onsite, tuple(self.basis.shells_per_atom[mask]))
        )

        # Train on-site along shell; Not train the padding zeros in on-site
        if train_onsite_type == "shell":
            y_data = [
                opt_onsite[..., ii]
                for ii in range(torch.max(self.basis.shells_per_atom))
            ]
            y_mask = [ii.ne(0) for ii in y_data]

            # Predict on-site for each orbitals
            onsite = (
                pack(
                    [
                        SciKitLearn(
                            self.geometry,
                            geometry_pred,
                            x_train=x_train[mask],
                            y_train=data.unsqueeze(-1)[mask],
                            x_test=x_test,
                            ml_method=ml_method,
                            form="atom",
                        ).prediction
                        for data, mask in zip(y_data, y_mask)
                    ]
                )
                .squeeze()
                .T
            )
        elif train_onsite_type == "element":
            uan = geometry_pred.unique_atomic_numbers
            an_train = self.geometry.atomic_numbers
            an_test = geometry_pred.atomic_numbers
            an_train = an_train[an_train.ne(0)]
            an_test = an_test[an_test.ne(0)]
            onsite = torch.zeros(len(an_test), opt_onsite.shape[-1])
            y_data = [opt_onsite[ii == an_train] for ii in uan]
            y_mask_train = [ii == an_train for ii in uan]
            y_mask_test = [ii == an_test for ii in uan]

            # Predict on-site for each element species
            _onsite = [
                SciKitLearn(
                    self.geometry,
                    geometry_pred,
                    x_train[mask_train],
                    opt_onsite[mask_train],  # Training data
                    x_test[mask_test],  # -> Testing input
                    ml_method=ml_method,
                    form="atom",
                ).prediction
                for mask_train, mask_test in zip(y_mask_train, y_mask_test)
            ]
            for ipred, mask in zip(_onsite, y_mask_test):
                onsite[mask] = ipred

        # basis_pred = Shell(geometry_pred.atomic_numbers, self.shell_dict)
        max_ls = torch.max(basis_pred.shells_per_atom)
        shells_per_atom = basis_pred.shells_per_atom[basis_pred.shells_per_atom.ne(0)]

        # Beyond max shell of the element, the onsite should be zero
        for ii in range(max_ls):
            onsite[..., ii][shells_per_atom < ii + 1] = 0

        return onsite[onsite.ne(0)]

    def _build_features(
        self, geometry_pred, y_train=None, feature_type="acsf", ml_method="nn", **kwargs
    ):
        # predict features
        x_train = Dscribe(self.geometry, feature_type=feature_type, **kwargs).features
        x_test = Dscribe(geometry_pred, feature_type=feature_type, **kwargs).features

        # use scikit learn to predict
        if y_train is not None:
            y_pred = SciKitLearn(
                self.geometry,
                geometry_pred,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                ml_method=ml_method,
            ).prediction
            return x_train, x_test, y_pred
        else:
            return x_train, x_test

    def __plot__(self, steps, loss, **kwargs):
        """Visualize training results."""
        compression_radii = kwargs.get("compression_radii", None)

        # plot loss
        plt.plot(np.linspace(1, steps, steps), loss)
        plt.ylabel("loss")
        plt.xlabel("steps")
        plt.show()

        # plot compression radii
        if compression_radii is not None:
            compr = pack(compression_radii)
            for ii in range(compr.shape[1]):
                for jj in range(compr.shape[2]):
                    plt.plot(np.linspace(1, steps, steps), compr[:, ii, jj])
            plt.show()

        # Add on-site for Hamiltonian
        if self.train_onsite == "local":
            ml_onsite = self.h_feed.on_site_dict["ml_onsite"].detach()
            plt.plot(self.onsite0, self.onsite0, "k")
            plt.plot(self.onsite0, ml_onsite, "rx", label="Onsite")
            plt.legend()
            plt.show()
        elif self.train_onsite == "global":
            for key, val in self.h_feed.on_site_dict.items():
                self.logger.info(f"onsite of {key}: {val.detach().tolist()}")

    def _dos(self, dos: Tensor, refdos: Tensor):
        """Construct loss of dos or pdos."""
        return hellinger(dos, refdos[..., 1])

    def _check(self, method: str):
        """Check the machine learning variables each step.

        When training compression radii, sometimes the compression radii will
        be out of range of given grid points and go randomly, therefore here
        the code makes sure the compression radii is in the defined range.
        """
        # detach remove initial graph and make sure compr_ml is leaf tensor
        if method in ("OptTvcr", "OptVcr"):
            if not self.global_r:
                compr = self.compr.detach().clone()
                min_mask = compr[compr != 0].lt(self.params["comp_r_min"])
                max_mask = compr[compr != 0].gt(self.params["comp_r_max"])
            else:
                vcr = self.compr0.detach().clone()
                min_mask = vcr[vcr != 0].lt(self.params["comp_r_min"])
                max_mask = vcr[vcr != 0].gt(self.params["comp_r_max"])

            if True in min_mask:
                if not self.global_r:
                    with torch.no_grad():
                        self.compr.clamp_(min=self.params["comp_r_min"])
                else:
                    with torch.no_grad():
                        self.compr0.clamp_(min=self.params["comp_r_min"])
            if True in max_mask:
                if not self.global_r:
                    with torch.no_grad():
                        self.compr.clamp_(max=self.params["comp_r_max"])
                else:
                    with torch.no_grad():
                        self.compr0.clamp_(min=self.params["comp_r_min"])

        # Constrain onsite in reasonable ranges
        if self.train_onsite == "local":
            with torch.no_grad():
                if (
                    (
                        self.h_feed.on_site_dict["ml_onsite"]
                        - (self.onsite0 + self.params["onsite_max"])
                    )
                    .gt(0)
                    .any()
                ):
                    self.logger.info("Onsite reach maximum limit")
                self.h_feed.on_site_dict["ml_onsite"].clamp_(
                    max=self.onsite0 + self.params["onsite_max"]
                )
            with torch.no_grad():
                if (
                    (
                        self.h_feed.on_site_dict["ml_onsite"]
                        - (self.onsite0 + self.params["onsite_min"])
                    )
                    .lt(0)
                    .any()
                ):
                    self.logger.info("Onsite reach minimum limit")
                self.h_feed.on_site_dict["ml_onsite"].clamp_(
                    min=self.onsite0 + self.params["onsite_min"]
                )
        elif self.train_onsite == "global":
            with torch.no_grad():
                for key, val in self.h_feed.on_site_dict.items():
                    if (
                        (val - (self.onsite0[key] + self.params["onsite_max"]))
                        .gt(0)
                        .any()
                    ):
                        self.logger.info("Onsite reach maximum limit")
                    self.h_feed.on_site_dict[key].clamp_(
                        max=self.onsite0[key] + self.params["onsite_max"]
                    )
            with torch.no_grad():
                for key, val in self.h_feed.on_site_dict.items():
                    if (
                        (val - (self.onsite0[key] + self.params["onsite_min"]))
                        .lt(0)
                        .any()
                    ):
                        self.logger.info("Onsite reach minimum limit")
                    self.h_feed.on_site_dict[key].clamp_(
                        min=self.onsite0[key] + self.params["onsite_min"]
                    )


class OptSpline(Model):
    """Optimize integrals with spline interpolation."""

    def __init__(
        self,
        geometry: Geometry,
        reference,
        path_to_skf: str,
        parameter,
        shell_dict,
        skf_type: Literal["h5", "skf"] = "h5",
        logger: logging.RootLogger = None,
        **kwargs,
    ):
        self.basis = Shell(geometry.atomic_numbers, shell_dict)
        self.shell_dict = shell_dict
        self.path_to_skf = path_to_skf
        self.skf_type = skf_type
        self.orbital_expand = kwargs.get("orbital_expand", True)
        self.train_onsite = kwargs.get("train_onsite", None)
        self.repulsive = kwargs.get("repulsive", False)
        build_abcd_h = kwargs.get("build_abcd_h", True)
        build_abcd_s = kwargs.get("build_abcd_s", True)
        global_onsite = kwargs.get("global_onsite", True)

        self.h_feed = SkfFeed.from_dir(
            path_to_skf,
            shell_dict,
            geometry=geometry,
            interpolation="Spline1d",
            integral_type="H",
            skf_type=skf_type,
            build_abcd=build_abcd_h,
            basis=self.basis,
            train_onsite=self.train_onsite,
            global_onsite=global_onsite,
            orbital_expand=self.orbital_expand,
        )
        self.s_feed = SkfFeed.from_dir(
            path_to_skf,
            shell_dict,
            skf_type=skf_type,
            geometry=geometry,
            interpolation="Spline1d",
            integral_type="S",
            build_abcd=build_abcd_s,
        )

        variables = []
        if build_abcd_h:
            variables.append({"params": self.h_feed.off_site_dict["variable"]})
        if build_abcd_s:
            variables.append({"params": self.s_feed.off_site_dict["variable"]})

        # Add on-site for Hamiltonian
        if self.train_onsite == "local":
            self.h_feed.on_site_dict["ml_onsite"].requires_grad_(True)
            variables.append(
                {
                    "params": self.h_feed.on_site_dict["ml_onsite"],
                    "lr": parameter["onsite_lr"],
                }
            )
            self.onsite0 = self.h_feed.on_site_dict["ml_onsite"].detach().clone()
        elif self.train_onsite == "global":
            self.onsite0 = {}
            for key, val in self.h_feed.on_site_dict.items():
                self.onsite0.update({key: val})
                val.requires_grad_(True)
                variables.append(
                    {
                        "params": self.h_feed.on_site_dict[key],
                        "lr": parameter["onsite_lr"],
                    }
                )

        super().__init__(geometry, reference, variables, parameter, logger, **kwargs)

        self.skparams = SkfParamFeed.from_dir(
            path_to_skf, self.geometry, skf_type=skf_type
        )
        if self.geometry.isperiodic:
            self.periodic = Periodic(
                self.geometry, self.geometry.cell, cutoff=self.skparams.cutoff, **kwargs
            )

    def forward(self, plot: bool = True, save: bool = True, **kwargs):
        """Train spline parameters with target properties."""
        super().forward()
        self._loss = []
        for istep in range(self.steps):
            self._update_train()
            self._loss.append(self.loss.detach().tolist())
            self.logger.info(f"step: {istep}, loss: %.6f" % self._loss[-1])

            break_tolerance = istep >= self.params["min_steps"]
            if self.reach_convergence and break_tolerance:
                break

        if plot:
            super().__plot__(istep + 1, self.loss_list[1:])

        return self.dftb

    def _update_train(self):
        if self.geometry.isperiodic:
            ham = hs_matrix(
                self.periodic,
                self.basis,
                self.h_feed,
                train_onsite=self.train_onsite,
                orbital_expand=self.orbital_expand,
            )
            over = hs_matrix(self.periodic, self.basis, self.s_feed)

        else:
            ham = hs_matrix(
                self.geometry,
                self.basis,
                self.h_feed,
                train_onsite=self.train_onsite,
                orbital_expand=self.orbital_expand,
            )
            over = hs_matrix(self.geometry, self.basis, self.s_feed)
        self.dftb = Dftb2(
            self.geometry,
            self.shell_dict,
            self.path_to_skf,
            from_skf=True,
            skf_type=self.skf_type,
        )
        self.dftb(hamiltonian=ham, overlap=over)
        super().__loss__(self.dftb)
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        super()._check("OptSpline")

    def predict(
        self, geometry_pred: object, ml_method: str = None, feature_type="acsf"
    ):
        """Predict with optimized Hamiltonian and overlap."""
        if ml_method is not None:
            self.ml_method = ml_method
        basis = Shell(geometry_pred.atomic_numbers, self.shell_dict)
        x_train, x_test = super()._build_features(
            geometry_pred, ml_method=self.ml_method, feature_type=feature_type
        )

        h_feed = SkfFeed.from_dir(
            self.path_to_skf,
            self.shell_dict,
            geometry=geometry_pred,
            interpolation="Spline1d",
            integral_type="H",
            build_abcd=False,
            basis=basis,
            train_onsite=self.train_onsite,
            orbital_expand=self.orbital_expand,
        )

        if self.train_onsite == "local":
            ml_onsite = super()._predict_onsite(
                geometry_pred, x_train, x_test, basis, self.ml_method
            )
            onsite_ref = h_feed.on_site_dict["ml_onsite"].detach().clone()
            if (onsite_ref - ml_onsite - self.params["onsite_max"]).gt(0).any():
                self.logger.info("Onsite reach maximum limit")
                ml_onsite.clamp_(max=onsite_ref + self.params["onsite_max"])
            if (onsite_ref - ml_onsite - self.params["onsite_min"]).lt(0).any():
                self.logger.info("Onsite reach minimum limit")
                ml_onsite.clamp_(min=onsite_ref + self.params["onsite_min"])
            self.h_feed.on_site_dict["ml_onsite"] = ml_onsite
        elif self.train_onsite == "global":
            for key, val in self.h_feed.on_site_dict.items():
                h_feed.on_site_dict[key] = self.h_feed.on_site_dict[key].detach()

        ham = hs_matrix(
            geometry_pred,
            basis,
            self.h_feed,
            train_onsite=self.train_onsite,
            orbital_expand=self.orbital_expand,
        )

        over = hs_matrix(geometry_pred, basis, self.s_feed)
        dftb2 = Dftb2(
            geometry=geometry_pred,
            shell_dict=self.shell_dict,
            path_to_skf=self.path_to_skf,
            repulsive=self.repulsive,
        )
        dftb2(hamiltonian=ham, overlap=over)
        return dftb2


class OptVcr(Model):
    """Optimize compression radii in basis functions."""

    def __init__(
        self,
        geometry: Geometry,
        reference,
        parameter,
        path_to_skf_vcr,
        path_to_skf_homo,
        vcr: Tensor,
        shell_dict: dict,
        skf_type: Literal["h5", "skf"] = "h5",
        logger: logging.RootLogger = None,
        **kwargs,
    ):
        self.vcr = vcr
        self.path_to_skf_vcr = path_to_skf_vcr
        self.path_to_skf_homo = path_to_skf_homo
        self.global_r = kwargs.get("global_r", False)
        self.repulsive = kwargs.get("repulsive", False)
        self.unique_atomic_numbers = geometry.unique_atomic_numbers
        self.orbital_expand = kwargs.get("orbital_expand", True)
        self.train_onsite = kwargs.get("train_onsite", None)

        if not self.global_r:
            self.compr = torch.zeros(*geometry.atomic_numbers.shape)
            init_dict = {
                1: torch.tensor([3.0]),
                6: torch.tensor([2.7]),
                7: torch.tensor([2.2]),
                8: torch.tensor([2.3]),
            }
            for ii, iu in enumerate(self.unique_atomic_numbers):
                mask = geometry.atomic_numbers == iu
                self.compr[mask] = init_dict[iu.tolist()]
            self.compr.requires_grad_(True)
            variables = [{"params": self.compr}]
        else:
            self.compr0 = torch.tensor([2.6, 2.2, 2.2, 2.4])
            self.compr = torch.zeros(geometry.atomic_numbers.shape)
            self.compr0.requires_grad_(True)
            variables = [{"params": self.compr0}]

        self.h_compr_feed = kwargs.get("h_compr_feed", True)
        self.s_compr_feed = kwargs.get("s_compr_feed", True)

        self.shell_dict = shell_dict
        self.basis = Shell(geometry.atomic_numbers, self.shell_dict)
        if self.h_compr_feed:
            self.h_feed = VcrFeed.from_dir(
                self.path_to_skf_vcr,
                self.shell_dict,
                vcr,
                skf_type="h5",
                geometry=geometry,
                integral_type="H",
                interpolation="BicubInterp",
                basis=self.basis,
                train_onsite=self.train_onsite,
                orbital_expand=self.orbital_expand,
            )
        if self.s_compr_feed:
            self.s_feed = VcrFeed.from_dir(
                self.path_to_skf_vcr,
                self.shell_dict,
                vcr,
                skf_type="h5",
                geometry=geometry,
                integral_type="S",
                interpolation="BicubInterp",
            )

        # Add on-site for Hamiltonian
        if self.train_onsite == "local":
            self.h_feed.on_site_dict["ml_onsite"].requires_grad_(True)
            variables.append(
                {
                    "params": self.h_feed.on_site_dict["ml_onsite"],
                    "lr": parameter["onsite_lr"],
                }
            )
            self.onsite0 = self.h_feed.on_site_dict["ml_onsite"].detach().clone()
        elif self.train_onsite == "global":
            self.onsite0 = {}
            for key, val in self.h_feed.on_site_dict.items():
                self.onsite0.update({key: val})
                val.requires_grad_(True)
                variables.append(
                    {
                        "params": self.h_feed.on_site_dict[key],
                        "lr": parameter["onsite_lr"],
                    }
                )

        if not self.global_r:
            super().__init__(
                geometry, reference, variables, parameter, logger, **kwargs
            )
        else:
            super().__init__(
                geometry, reference, variables, parameter, logger, **kwargs
            )

        self.skparams = SkfParamFeed.from_dir(
            self.path_to_skf_homo,
            self.geometry,
            skf_type=skf_type,
            repulsive=self.repulsive,
        )
        if self.geometry.isperiodic:
            self.periodic = Periodic(
                self.geometry, self.geometry.cell, cutoff=self.skparams.cutoff, **kwargs
            )
        else:
            self.periodic = None

    def forward(self, plot: bool = True, save: bool = True, **kwargs):
        """Train compression radii with target properties."""
        super().forward()
        self._compr = []
        self.ham_list, self.over_list = [], []
        self._loss = []
        for istep in range(self.steps):
            self._update_train()
            self._loss.append(self.loss.detach().tolist())
            self.logger.info(f"step: {istep}, loss: %.6f" % self._loss[-1])

            break_tolerance = istep >= self.params["min_steps"]
            if self.reach_convergence and break_tolerance:
                break

        if plot:
            super().__plot__(
                istep + 1, self.loss_list[1:], compression_radii=self._compr
            )
        return self.dftb

    def _update_train(self):
        if self.global_r:
            for ii, iu in enumerate(self.unique_atomic_numbers):
                mask = self.geometry.atomic_numbers == iu
                self.compr[mask] = self.compr0[ii]

        hs_obj = self.periodic if self.geometry.isperiodic else self.geometry
        if self.h_compr_feed:
            ham = hs_matrix(
                hs_obj,
                self.basis,
                self.h_feed,
                multi_varible=self.compr,
                train_onsite=self.train_onsite,
                orbital_expand=self.orbital_expand,
            )
        else:
            ham = hs_matrix(
                hs_obj,
                self.basis,
                self.h_feed2,
                train_onsite=self.train_onsite,
                orbital_expand=self.orbital_expand,
            )

        if self.s_compr_feed:
            over = hs_matrix(hs_obj, self.basis, self.s_feed, multi_varible=self.compr)
        else:
            over = hs_matrix(hs_obj, self.basis, self.s_feed2)

        self.ham_list.append(ham.detach()), self.over_list.append(over.detach())
        self.dftb = Dftb2(
            self.geometry, self.shell_dict, self.path_to_skf_homo, self.repulsive
        )
        self.dftb(hamiltonian=ham, overlap=over)
        super().__loss__(self.dftb)
        self._compr.append(self.compr.detach().clone())
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self._check("OptVcr")

    def predict(
        self,
        geometry_pred: object,
        ml_method: str = None,
        feature_type="acsf",
        **kwargs,
    ):
        """Predict with optimized Hamiltonian and overlap."""
        if ml_method is not None:
            self.ml_method = ml_method
        basis_pred = Shell(geometry_pred.atomic_numbers, self.shell_dict)

        # predict features
        y_train = self.compr.detach()[self.geometry.atomic_numbers.ne(0)].unsqueeze(-1)
        x_train, x_test, y_pred = super()._build_features(
            geometry_pred, y_train, feature_type, ml_method, **kwargs
        )
        y_pred.clamp_(min=self.params["comp_r_min"])
        y_pred.clamp_(max=self.params["comp_r_max"])
        y_pred = y_pred.squeeze(-1)

        h_feed_pred = VcrFeed.from_dir(
            self.path_to_skf_vcr,
            self.shell_dict,
            self.vcr,
            geometry=geometry_pred,
            basis=basis_pred,
            skf_type="h5",
            integral_type="H",
            interpolation="BicubInterp",
            train_onsite=self.train_onsite,
            orbital_expand=self.orbital_expand,
        )
        s_feed_pred = VcrFeed.from_dir(
            self.path_to_skf_vcr,
            self.shell_dict,
            self.vcr,
            geometry=geometry_pred,
            basis=basis_pred,
            skf_type="h5",
            integral_type="S",
            interpolation="BicubInterp",
        )

        # Replace onsite from ML predictions
        if self.train_onsite == "local":
            ml_onsite = super()._predict_onsite(
                geometry_pred, x_train, x_test, basis_pred, self.ml_method
            )

            onsite_ref = h_feed_pred.on_site_dict["ml_onsite"].clone()
            if (onsite_ref - ml_onsite - self.params["onsite_max"]).gt(0).any():
                self.logger.info("Onsite reach maximum limit")
                ml_onsite.clamp_(max=onsite_ref + self.params["onsite_max"])
            if (onsite_ref - ml_onsite - self.params["onsite_min"]).lt(0).any():
                self.logger.info("Onsite reach minimum limit")
                ml_onsite.clamp_(min=onsite_ref + self.params["onsite_min"])
            h_feed_pred.on_site_dict["ml_onsite"] = ml_onsite
            plt.plot(onsite_ref, onsite_ref, "k")
            plt.plot(onsite_ref, ml_onsite, "bx")
            plt.show()
            plt.hist(onsite_ref - ml_onsite, 20, facecolor='g', alpha=0.75)
            plt.xlabel("difference between optimized and original on-site")
            plt.ylabel('counts')
            plt.show()
        elif self.train_onsite == "global":
            for key, val in self.h_feed.on_site_dict.items():
                h_feed_pred.on_site_dict[key] = self.h_feed.on_site_dict[key].detach()

        if self.h_compr_feed:
            ham = hs_matrix(
                geometry_pred,
                basis_pred,
                h_feed_pred,
                multi_varible=y_pred,
                train_onsite=self.train_onsite,
                orbital_expand=self.orbital_expand,
            )
        else:
            ham = hs_matrix(geometry_pred, basis_pred, h_feed_pred)
        if self.s_compr_feed:
            over = hs_matrix(
                geometry_pred, basis_pred, s_feed_pred, multi_varible=y_pred
            )
        else:
            over = hs_matrix(geometry_pred, basis_pred, s_feed_pred)

        dftb2 = Dftb2(
            geometry=geometry_pred,
            shell_dict=self.shell_dict,
            path_to_skf=self.path_to_skf_homo,
            repulsive=self.repulsive,
        )
        dftb2()
        plt.plot(dftb2.ham.flatten(), dftb2.ham.flatten(), "k")
        plt.plot(dftb2.ham.flatten(), ham.flatten(), "rx")
        plt.show()
        dftb2(hamiltonian=ham, overlap=over)

        return dftb2


class OptTvcr(Model):
    """Optimize two center integrals with two different compression radii.

    This class allows you both training the compression radii in two-center
    integrals or training the on-site energy, globally or locally. For
    two-center integrals, we design a multi-varibles interpolation which
    could generate the gradients for backpropagation.

    Arguments:
        geometry: Geometry object with training data.
        reference: Reference data for training, usually from DFT.
        path_to_skf_tvcr: Path to binary Slater-Koster files with two
            different compression radii.
        parameter: Parameters dictionary for machine learning.
        shell_dict: Shell dictionary for each element species.
        logger: Logger object.

    """

    def __init__(
        self,
        geometry: Geometry,
        reference: Dict,
        path_to_skf_tvcr: str,
        parameter: Dict,
        tvcr: Tensor,
        shell_dict: Dict,
        logger: logging.RootLogger = None,
        **kwargs,
    ):
        """Initialize parameters."""
        self.orbital_expand = kwargs.get("orbital_expand", True)
        self.interpolation = kwargs.get("interpolation", "MultiVarInterp")
        self.train_onsite = kwargs.get("train_onsite", None)
        self.path_to_skf = path_to_skf_tvcr
        self.tvcr = tvcr
        self.global_r = kwargs.get("global_r", False)
        self.unique_atomic_numbers = geometry.unique_atomic_numbers
        self.repulsive = kwargs.get("repulsive", False)
        self.periodic = kwargs.get("periodic") if geometry.isperiodic else None
        self.klines = kwargs.get("klines", None)
        self.h_compr_feed = kwargs.get("h_compr_feed", True)
        self.s_compr_feed = kwargs.get("s_compr_feed", True)
        self.shell_dict = shell_dict
        self.basis = Shell(geometry.atomic_numbers, self.shell_dict)

        # Start from mio initial guess for both global and local radii
        if not self.global_r:
            self.compr = torch.zeros(*geometry.atomic_numbers.shape, 2)
            init_dict = {
                1: torch.tensor([2.5, 3.0]),
                6: torch.tensor([7.0, 2.7]),
                7: torch.tensor([8.0, 2.2]),
                8: torch.tensor([8.0, 2.3]),
                14: torch.tensor([3.3, 7.6]),
            }
            for ii, iu in enumerate(self.unique_atomic_numbers):
                mask = geometry.atomic_numbers == iu
                self.compr[mask] = init_dict[iu.tolist()]
            self.compr.requires_grad_(True)
            variables = [{"params": self.compr}]
        else:
            self.compr = torch.zeros(*geometry.atomic_numbers.shape, 2)
            self.compr0 = torch.tensor(
                [[2.5, 3.0], [7.0, 2.7], [8.0, 2.2], [8.0, 2.3]]
            ).requires_grad_(True)
            variables = [{"params": self.compr0}]

        # Get Hamiltonian Feed from input, both one-center integrals
        # and two-center integrals could be trained
        if self.h_compr_feed:
            self.h_feed = TvcrFeed.from_dir(
                path_to_skf_tvcr,
                self.shell_dict,
                tvcr,
                geometry=geometry,
                basis=self.basis,
                skf_type="h5",
                integral_type="H",
                interpolation=self.interpolation,
                train_onsite=self.train_onsite,
                orbital_expand=self.orbital_expand,
            )

            # Add on-site for Hamiltonian
            if self.train_onsite == "local":
                self.h_feed.on_site_dict["ml_onsite"].requires_grad_(True)
                variables.append(
                    {
                        "params": self.h_feed.on_site_dict["ml_onsite"],
                        "lr": parameter["onsite_lr"],
                    }
                )
                self.onsite0 = self.h_feed.on_site_dict["ml_onsite"].detach().clone()
            elif self.train_onsite == "global":
                self.onsite0 = {}
                for key, val in self.h_feed.on_site_dict.items():
                    self.onsite0.update({key: val})
                    val.requires_grad_(True)
                    variables.append(
                        {
                            "params": self.h_feed.on_site_dict[key],
                            "lr": parameter["onsite_lr"],
                        }
                    )

        # Get Overlap Feed from input, only train two-center integrals
        if self.s_compr_feed:
            self.s_feed = TvcrFeed.from_dir(
                path_to_skf_tvcr,
                self.shell_dict,
                tvcr,
                geometry=geometry,
                skf_type="h5",
                integral_type="S",
                interpolation=self.interpolation,
            )

        if not self.global_r:
            super().__init__(
                geometry, reference, variables, parameter, logger, **kwargs
            )
        else:
            super().__init__(
                geometry, reference, variables, parameter, logger, **kwargs
            )

    def forward(
        self,
        break_tolerance: bool = True,  # Stop if converge
        plot: bool = True,  # -> Plot the Loss and training results
        save: bool = True,  # -> Save and make persistence
        **kwargs,
    ):
        """Train compression radii with target properties."""
        super().forward()
        self._compr = []
        self.ham_list, self.over_list = [], []
        for istep in range(self.steps):
            self._update_train(**kwargs)
            print("self.compr", self.compr)
            self.logger.info(
                "step: " + str(istep) + " loss: " + str(self.loss.detach().tolist())
            )

            break_tolerance = istep >= self.params["min_steps"]
            if self.reach_convergence and break_tolerance:
                break

        if plot:
            super().__plot__(
                istep + 1, self.loss_list[1:], compression_radii=self._compr
            )

        return self.dftb

    def _update_train(self, **kwargs):
        """Update parameters for each training steps."""
        if self.global_r:
            for ii, iu in enumerate(self.unique_atomic_numbers):
                mask = self.geometry.atomic_numbers == iu
                self.compr[mask] = self.compr0[ii]

        geo = self.periodic if self.geometry.isperiodic else self.geometry
        if self.h_compr_feed:
            ham = hs_matrix(
                geo,
                self.basis,
                self.h_feed,
                multi_varible=self.compr,
                train_onsite=self.train_onsite,
                orbital_expand=self.orbital_expand,
            )
        else:
            ham = hs_matrix(geo, self.basis, self.h_feed2)

        if self.s_compr_feed:
            over = hs_matrix(geo, self.basis, self.s_feed, multi_varible=self.compr)
        else:
            over = hs_matrix(geo, self.basis, self.s_feed2)

        self.ham_list.append(ham.detach()), self.over_list.append(over.detach())
        self.dftb = Dftb1(
            geometry=self.geometry,
            shell_dict=self.shell_dict,
            path_to_skf=self.path_to_skf,
            skf_type="h5",
            repulsive=self.repulsive,
            klines=self.klines,
        )
        self.dftb(hamiltonian=ham, overlap=over)
        super().__loss__(self.dftb)
        self._compr.append(self.compr.detach().clone())
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        super()._check("OptTvcr")

    def predict(self, geometry_pred: object, ml_method: str = None, **kwargs):
        """Predict with optimized Hamiltonian and overlap."""
        if ml_method is not None:
            self.ml_method = ml_method
        basis_pred = Shell(geometry_pred.atomic_numbers, self.shell_dict)

        # # predict features
        feature_type = "acsf"

        # use scikit learn to predict
        y_train = self.compr.detach()[self.geometry.atomic_numbers.ne(0)]
        x_train, x_test, y_pred = super()._build_features(
            geometry_pred, y_train, feature_type, ml_method, **kwargs
        )
        y_pred.clamp_(min=self.params["comp_r_min"])
        y_pred.clamp_(max=self.params["comp_r_max"])

        h_feed_pred = TvcrFeed.from_dir(
            self.path_to_skf,
            self.shell_dict,
            self.tvcr,
            geometry=geometry_pred,
            basis=basis_pred,
            skf_type="h5",
            integral_type="H",
            interpolation=self.interpolation,
            train_onsite=self.train_onsite,
            orbital_expand=self.orbital_expand,
        )
        s_feed_pred = TvcrFeed.from_dir(
            self.path_to_skf,
            self.shell_dict,
            self.tvcr,
            geometry=geometry_pred,
            basis=basis_pred,
            skf_type="h5",
            integral_type="S",
            interpolation=self.interpolation,
        )

        # Replace onsite from ML predictions
        if self.train_onsite == "local":
            ml_onsite = super()._predict_onsite(
                geometry_pred, x_train, x_test, basis_pred, self.ml_method
            )
            onsite_ref = h_feed_pred.on_site_dict["ml_onsite"].clone()
            if (onsite_ref - ml_onsite - self.params["onsite_max"]).gt(0).any():
                self.logger.info("Onsite reach maximum limit")
                ml_onsite.clamp_(max=onsite_ref + self.params["onsite_max"])
            if (onsite_ref - ml_onsite - self.params["onsite_min"]).lt(0).any():
                self.logger.info("Onsite reach minimum limit")
                ml_onsite.clamp_(min=onsite_ref + self.params["onsite_min"])
            h_feed_pred.on_site_dict["ml_onsite"] = ml_onsite

        if self.h_compr_feed:
            ham = hs_matrix(
                geometry_pred,
                basis_pred,
                h_feed_pred,
                multi_varible=y_pred,
                train_onsite=self.train_onsite,
                orbital_expand=self.orbital_expand,
            )
        else:
            ham = hs_matrix(geometry_pred, basis_pred, h_feed_pred)
        if self.s_compr_feed:
            over = hs_matrix(
                geometry_pred, basis_pred, s_feed_pred, multi_varible=y_pred
            )
        else:
            over = hs_matrix(geometry_pred, basis_pred, s_feed_pred)

        dftb2 = Dftb2(
            geometry=geometry_pred,
            shell_dict=self.shell_dict,
            path_to_skf=self.path_to_skf,
            repulsive=self.repulsive,
        )
        dftb2(hamiltonian=ham, overlap=over)
        return dftb2

    def __repr__(self):
        """Return representation of class `OptTvcr`."""
        n_batch = self.geometry.atomic_numbers.shape[0]
        return "OptTvcr " + str(n_batch) + " batch size"
