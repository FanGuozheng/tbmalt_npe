#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example to run training."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from tbmalt import Geometry
from tbmalt.ml.optim import OptSpline, OptVcr, OptTvcr
from tbmalt.io.hdf import LoadHdf
from tbmalt.physics.dftb.dftb import Dftb2

torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)


###################
# optimize params #
###################
target = "train"
device = torch.device("cpu")
size_opt = 1000
opt_type = "vcr"
ani = 3
global_r = False
parallel = True
h_compr_feed = True
s_compr_feed = True
break_tolerance = True
choice = "random"
# If expand P, D orbitals to 3, 5...
orbital_expand = False
if ani == 1:
    dataset_aims = "./dataset/aims_6000_01.hdf"
    dataset_dftb = "./dataset/scc_6000_01.hdf"
elif ani == 3:
    dataset_aims = "./dataset/aims_2000_03.hdf"
    dataset_dftb = "./dataset/scc_2000_03.hdf"

# ML parameters
ml_params = {
    "targets": ["cpa"],  # charge, dipole, gap, cpa, homo_lumo
    "loss_function": "MSELoss",
    "optimizer": "Adam",
    "ml_method": "random_forest",  # nn, linear, random_forest
    "min_steps": 6,
    "max_steps": 200,
    "tolerance": 1e-6,
    "lr": 0.02,
    "onsite_lr": 5e-4,  # lr for onsite
    "charge_weight": 1.0,
    "dipole_weight": 1.0,
    "cpa_weight": 1.0,
    "comp_r_min": 2.0,  # Min limit of R
    "comp_r_max": 9.5,  # Max limit of R
    "onsite_min": -0.1,  # Max shift of onsite
    "onsite_max": 0.1,
}


shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
vcr = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0])

##################
# predict params #
##################
# Set target = 'test' to start predict, otherwise params will be not used
ani_pred = 3  # decide which file to load
size_pred = 400
test_ratio = 1  # Select from all dataset, could be trained data
pickle_file = "/mnt/local/gz_fan_data/work/train_data/data/opt/vcr_1000ani3_onsite_local_dipole_data1.pkl"

if ani_pred == 1:
    dataset_aims_pred = "./dataset/aims_6000_01.hdf"
    dataset_dftb_pred = "./dataset/scc_6000_01.hdf"
elif ani_pred == 3:
    dataset_aims_pred = "./dataset/aims_6000_03.hdf"
    dataset_dftb_pred = "./dataset/scc_6000_03.hdf"
elif ani_pred == 5:
    dataset_aims_pred = "./dataset/aims_6000_05.hdf"
    dataset_dftb_pred = "./dataset/scc_20000_05.hdf"
elif ani_pred == 7:
    dataset_aims_pred = "./dataset/aims_6000_07.hdf"
    dataset_dftb_pred = "./dataset/scc_20000_07.hdf"


def train(dataset_ref, size, dataset_dftb=None, **kwargs):
    """Optimize spline parameters or compression radii."""
    geo_opt, data_ref = _load_ref(
        dataset_ref, size, ["charge", "dipole", "hirshfeld_volume_ratio"]
    )
    data_ref["cpa"] = data_ref["hirshfeld_volume_ratio"]
    if dataset_dftb is not None:
        geo_dftb, data_dftb = _load_ref(dataset_dftb, size, ["charge", "dipole"])

    # optimize integrals with spline parameters
    if opt_type == "spline":
        path_to_skf = "./slko/mio_new.hdf"
        train_onsite = "global"
        ml_params.update({"lr": 0.001, "onsite_lr": 2e-6})
        opt = OptSpline(
            geometry=geo_opt,
            reference=data_ref,
            path_to_skf=path_to_skf,
            parameter=ml_params,
            shell_dict=shell_dict,
            train_onsite=train_onsite,
            orbital_expand=orbital_expand,
        )

    # optimize integrals with compression radii
    elif opt_type == "vcr":
        path_to_skf = "./vcr.h5"
        path_to_skf2 = "./slko/mio_new.hdf"
        train_onsite = "global" if global_r else "local"
        opt = OptVcr(
            geometry=geo_opt,
            reference=data_ref,
            path_to_skf_vcr=path_to_skf,
            path_to_skf_homo=path_to_skf2,
            parameter=ml_params,
            vcr=vcr,
            shell_dict=shell_dict,
            h_compr_feed=h_compr_feed,
            s_compr_feed=s_compr_feed,
            interpolation="BicubInterp",
            global_r=global_r,
            train_onsite=train_onsite,
            orbital_expand=orbital_expand,
        )

    # optimize integrals with compression radii
    elif opt_type == "tvcr":
        tvcr = torch.tensor([2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
        path_to_skf = "./tvcr.h5"
        # train_onsite = "global" if global_r else "local"
        opt = OptTvcr(
            geometry=geo_opt,
            reference=data_ref,
            path_to_skf_tvcr=path_to_skf,
            parameter=ml_params,
            tvcr=tvcr,
            shell_dict=shell_dict,
            h_compr_feed=h_compr_feed,
            s_compr_feed=s_compr_feed,
            interpolation="BSpline",
            # global_r=global_r,
            train_onsite=train_onsite,
            orbital_expand=orbital_expand,
        )

    # If running in parallel
    if parallel:
        opt_parallel = torch.nn.DataParallel(opt, device_ids=[0, 1, 2, 3, 4, 5])
        dftb = opt_parallel(break_tolerance=break_tolerance)
    else:
        dftb = opt(break_tolerance=break_tolerance)

    # save training instance
    pkl_name = (
        opt_type
        + "_"
        + str(size_opt)
        + "ani"
        + str(ani)
        + "_onsite_"
        + train_onsite
        + "_"
        + "_".join(ml_params["targets"])
        + ".pkl"
    )

    with open(pkl_name, "wb") as f:
        pickle.dump(opt, f)

    dftb_mio = _cal_cpa(geo_opt, path="./slko/mio_new.hdf")
    if "cpa" in ml_params["targets"]:
        _plot(data_ref["cpa"], dftb.cpa, dftb_mio.cpa, "cpa")

    # Plot charge and dipole in any cases
    for target in ["charge", "dipole"]:
        if target not in ml_params["targets"]:
            ml_params["targets"].append(target)
    for target in ml_params["targets"]:
        mio_data = getattr(dftb_mio, target)
        _plot(data_ref[target], getattr(dftb, target), mio_data, target)
        if target == "charge":
            qref = data_ref[target] - getattr(dftb, "qzero")
            qopt = getattr(dftb, target) - getattr(dftb, "qzero")
            qmio = getattr(dftb_mio, target) - getattr(dftb, "qzero")
            _plot(qref, qopt, qmio, "delta_q")


def test(pickle_file: str, size: int, **kwargs):
    """Test optimized results."""
    # load optimized object
    loaded_model = pickle.load(open(pickle_file, "rb"))

    # update training params in optimized object
    ml_params["targets"] = ["charge", "dipole"]
    loaded_model.params["ml_method"] = ml_params["ml_method"]

    for target in ml_params["targets"]:
        plt.plot(loaded_model.reference[target], loaded_model.reference[target], "k")
        plt.plot(
            loaded_model.reference[target],
            getattr(loaded_model.dftb, target).detach(),
            "rx",
        )
        plt.show()

    _, data_dftb = _load_ref(
        dataset_dftb_pred, size, ["charge", "dipole"], test_ratio=test_ratio
    )
    geo_aims, data_aims = _load_ref(
        dataset_aims_pred,
        size,
        ["charge", "dipole", "hirshfeld_volume_ratio"],
        test_ratio=test_ratio,
    )
    dftb = loaded_model.predict(
        geo_aims, ml_method=ml_params["ml_method"], feature_type="acsf"
    )

    dftb_mio = _cal_cpa(geo_aims, path="./slko/mio_new.hdf")
    _plot(data_aims["hirshfeld_volume_ratio"], dftb.cpa, dftb_mio.cpa, "cpa")

    for target in ml_params["targets"]:
        mio_data = getattr(dftb_mio, target)
        _plot(data_aims[target], getattr(dftb, target), mio_data, target)


def _load_ref(dataset, size, properties, units="angstrom", **kwargs):
    """Helper function to load dataset, return `Geometry` object, data."""
    test_ratio = kwargs.get("test_ratio", 1.0)
    numbers, positions, data = LoadHdf.load_reference(
        dataset, size, properties, test_ratio=test_ratio, choice=choice
    )
    cell = kwargs.get("cell", None)
    geo = Geometry(numbers, positions, units=units, cell=cell)

    return geo, data


def _cal_cpa(geometry, path):
    dftb = Dftb2(geometry, shell_dict, path, from_skf=True)
    dftb()
    return dftb


def _plot(
    reference: Tensor, data1: Tensor, data2: Tensor, target: str, save: bool = True
):
    """Plot single target with optimized value and DFTB value."""
    mae1 = (abs(reference - data1)).sum() / reference.shape[0]
    print(
        "(abs(reference - data1)).sum()",
        (abs(reference - data1)).sum(),
        reference.shape,
    )
    mae2 = (abs(reference - data2)).sum() / reference.shape[0]
    print(
        "(abs(reference - data1)).sum()",
        (abs(reference - data2)).sum(),
        reference.shape,
    )
    rmin, rmax = torch.min(reference), torch.max(reference)
    plt.plot(np.linspace(rmin, rmax), np.linspace(rmin, rmax), "k")
    plot1 = plt.plot(reference, data1.detach(), "rx")
    plot2 = plt.plot(reference, data2.detach(), "bo")
    plt.xlabel(f"reference {target}")
    plt.ylabel(f"optimized {target} and DFTB {target}")
    plt.legend([plot1[0], plot2[0]], [f"opt MAE: {mae1}", f"DFTB MAE: {mae2}"])
    if save:
        plt.savefig(target, dpi=300)
    plt.show()


if __name__ == "__main__":
    """Main function."""
    if target == "train":
        train(dataset_aims, size_opt, dataset_dftb, device=device)
    elif target == "test":
        test(pickle_file, size_pred, device=device)
