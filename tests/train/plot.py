#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:50:32 2022

@author: gz_fan
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FormatStrFormatter

from tbmalt import Geometry, Dftb2
from tbmalt.io.hdf import LoadHdf
from tbmalt import SkfFeed, Shell, hs_matrix

torch.set_default_dtype(torch.float64)

size_pred, test_ratio = 400, 1  # Chooese 400 molecules for testing
version = 'old'
dataset_aims1 = "./dataset/aims_6000_01.hdf"
dataset_dftb1 = "./dataset/scc_6000_01.hdf"
dataset_aims3 = "./dataset/aims_20000_03.hdf"
dataset_dftb3 = "./dataset/scc_20000_03.hdf"
target = ["charge", "dipole"]
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}


def pred_dip_size():
    """Predictions of two different ML method with different training size."""
    size_opt = [200, 400, 600, 800, 1000, 1200, 1400]

    # Global optmization, equals to standard DFTB parametrizations
    comp_gr_train_err1 = [
        [0.0277, 0.0302, 0.0285, 0.0292, 0.0300, 0.0299, 0.0297],
        [0.0295, 0.0298, 0.0295, 0.0293, 0.0294, 0.0292, 0.0288],
        [0.0324, 0.0314, 0.0292, 0.0296, 0.0297, 0.0302, 0.0294],
    ]
    comp_gr_pred_err1 = torch.tensor(
        [  # 200     400     600      800    1000    1200    1400
            [0.0300, 0.0296, 0.0290, 0.0309, 0.0295, 0.0288, 0.0296],
            [0.0292, 0.0299, 0.0293, 0.0304, 0.0300, 0.0306, 0.0288],
            [0.0299, 0.0307, 0.0301, 0.0298, 0.0292, 0.0291, 0.0295],
        ]
    )
    avg_pred_comp_gr_err1 = comp_gr_pred_err1.sum(0) / comp_gr_pred_err1.shape[0]
    pred_gr_err1_min = torch.abs(
        torch.min(comp_gr_pred_err1 - avg_pred_comp_gr_err1.unsqueeze(0), 0)[0]
    )
    pred_gr_err1_max = torch.abs(
        torch.max(comp_gr_pred_err1 - avg_pred_comp_gr_err1, 0)[0]
    )

    # Local optmization, ML DFTB basis parameters
    comp_lr_train_err1 = [
        [0.0070, 0.0070, 0.0070, 0.0067, 0.0069, 0.0067, 0.0070],
        [0.0064, 0.0068, 0.0071, 0.0068, 0.0068, 0.0068, 0.0069],
        [0.0070, 0.0067, 0.0067, 0.0068, 0.0066, 0.0069, 0.0068],
    ]
    comp_lr_pred_err1 = torch.tensor(
        [
            # 200     400     600      800    1000    1200    1400
            [0.0187, 0.0165, 0.0152, 0.0150, 0.0142, 0.0140, 0.0134],
            [0.0189, 0.0168, 0.0158, 0.0153, 0.0142, 0.0139, 0.0138],
            [0.0182, 0.0167, 0.0154, 0.0148, 0.0142, 0.0142, 0.0134],
        ]
    )
    avg_pred_comp_lr_err1 = comp_lr_pred_err1.sum(0) / comp_lr_pred_err1.shape[0]
    pred_lr_err1_min = torch.abs(
        torch.min(comp_lr_pred_err1 - avg_pred_comp_lr_err1.unsqueeze(0), 0)[0]
    )
    pred_lr_err1_max = torch.abs(
        torch.max(comp_lr_pred_err1 - avg_pred_comp_lr_err1, 0)[0]
    )

    int_pred_err1 = torch.tensor(
        [  # 200     400     600      800    1000    1200    1400
            [0.0752, 0.0498, 0.0418, 0.0436, 0.0378, 0.0328, 0.0349],
            [0.0775, 0.0552, 0.0414, 0.0387, 0.0354, 0.0353, 0.0342],
            [0.0742, 0.0516, 0.0441, 0.0401, 0.0350, 0.0354, 0.0348],
        ]
    )
    avg_pred_int_err1 = int_pred_err1.sum(0) / int_pred_err1.shape[0]
    pred_int_err1_min = torch.abs(
        torch.min(int_pred_err1 - avg_pred_int_err1.unsqueeze(0), 0)[0]
    )
    pred_int_err1_max = torch.abs(torch.max(int_pred_err1 - avg_pred_int_err1, 0)[0])

    # Global optmization, equals to standard DFTB parametrizations
    comp_gr_pred_err3 = torch.tensor(
        [
            # 200     400     600      800    1000    1200    1400
            [0.2848, 0.2811, 0.2459, 0.2452, 0.2511, 0.2794, 0.2715],
            [0.2814, 0.2826, 0.2492, 0.2290, 0.2794, 0.2548, 0.2716],
            [0.2649, 0.2609, 0.2860, 0.2848, 0.2807, 0.2806, 0.2730],
            [0.2437, 0.2508, 0.2834, 0.2771, 0.2538, 0.2511, 0.2680],
            [0.2782, 0.2868, 0.2660, 0.2932, 0.2768, 0.2766, 0.2774],
            [0.2628, 0.2542, 0.2841, 0.2778, 0.2827, 0.2731, 0.2650],
            [0.2696, 0.2825, 0.2535, 0.2829, 0.2399, 0.2792, 0.2677],
        ]
    )
    avg_pred_comp_gr_err3 = comp_gr_pred_err3.sum(0) / comp_gr_pred_err3.shape[0]
    pred_gr_err3_min = torch.abs(
        torch.min(comp_gr_pred_err3 - avg_pred_comp_gr_err3.unsqueeze(0), 0)[0]
    )
    pred_gr_err3_max = torch.abs(
        torch.max(comp_gr_pred_err3 - avg_pred_comp_gr_err3, 0)[0]
    )

    # Local optmization, ML DFTB basis parameters
    comp_lr_pred_err3 = torch.tensor(
        [  # 200     400     600      800    1000    1200    1400
            [0.1271, 0.1163, 0.1089, 0.1027, 0.1005, 0.0989, 0.0984],
            [0.1267, 0.1124, 0.1106, 0.1004, 0.1007, 0.0947, 0.1007],
            [0.1298, 0.1105, 0.1075, 0.0986, 0.0985, 0.0936, 0.0984],
        ]
    )
    avg_pred_comp_lr_err3 = comp_lr_pred_err3.sum(0) / comp_lr_pred_err3.shape[0]
    pred_lr_err3_min = torch.abs(
        torch.min(comp_lr_pred_err3 - avg_pred_comp_lr_err3.unsqueeze(0), 0)[0]
    )
    pred_lr_err3_max = torch.abs(
        torch.max(comp_lr_pred_err3 - avg_pred_comp_lr_err3, 0)[0]
    )

    # Spline model
    int_pred_err3 = torch.tensor(
        [  # 200     400     600      800    1000    1200    1400
            [0.2614, 0.2367, 0.2197, 0.2056, 0.2210, 0.2032, 0.2031],
            [0.2445, 0.2234, 0.2162, 0.2400, 0.2024, 0.2136, 0.2466],
            [0.2516, 0.2253, 0.2065, 0.2255, 0.2009, 0.2319, 0.2544],
            [0.2254, 0.2229, 0.2183, 0.1977, 0.2120, 0.2097, 0.2025],
            [0.2572, 0.2633, 0.2244, 0.2234, 0.2206, 0.2034, 0.2056],
            [0.2220, 0.2344, 0.2199, 0.2255, 0.2103, 0.2095, 0.2121],
            [0.2449, 0.2326, 0.2376, 0.2208, 0.2170, 0.2206, 0.2066],
        ]
    )
    avg_pred_int_err3 = int_pred_err3.sum(0) / int_pred_err3.shape[0]
    pred_int_err3_min = torch.abs(
        torch.min(int_pred_err3 - avg_pred_int_err3.unsqueeze(0), 0)[0]
    )
    pred_int_err3_max = torch.abs(torch.max(int_pred_err3 - avg_pred_int_err3, 0)[0])
    color = "tab:red"
    plt.errorbar(
        size_opt,
        avg_pred_int_err1,
        color=color,
        yerr=torch.stack([pred_int_err1_min, pred_int_err1_max]),
        marker="o",
        label="spline",  # + r"$ANI-1_1$",
    )
    plt.errorbar(
        size_opt,
        avg_pred_int_err3,
        yerr=torch.stack([pred_int_err3_min, pred_int_err3_max]),
        color=color,
        fmt="--o",
    )
    color = "tab:blue"
    plt.errorbar(
        size_opt,
        avg_pred_comp_lr_err1,
        color=color,
        fmt="-v",
        yerr=torch.stack([pred_lr_err1_min, pred_lr_err1_max]),
        label="local basis functions",  # + r"$ANI-1_1$",
    )
    plt.errorbar(
        size_opt,
        avg_pred_comp_lr_err3,
        color=color,
        fmt="--v",
        yerr=torch.stack([pred_lr_err3_min, pred_lr_err3_max]),
    )
    color = "tab:cyan"
    plt.errorbar(
        size_opt,
        avg_pred_comp_gr_err1,
        color=color,
        yerr=torch.stack([pred_gr_err1_min, pred_gr_err1_max]),
        fmt="-s",
        label="global basis functions",  # + r"$ANI-1_1$",
    )
    plt.errorbar(
        size_opt,
        avg_pred_comp_gr_err3,
        color=color,
        yerr=torch.stack([pred_gr_err3_min, pred_gr_err3_max]),
        fmt="--s",
    )
    plt.xlabel("Traning dataset size", fontsize="large")
    plt.ylim(0.0, 0.38)
    plt.ylabel("Testing MAE of dipole (au)", fontsize="large")
    plt.legend(fontsize="medium", edgecolor=None)
    plt.savefig("sizeDip.png", dpi=300)
    plt.show()


def other_pro(
    comp_lr="./data/opt/vcr_1000ani1_onsite_local_dipole.pkl",
    comp_gr="./data/opt/vcr_1000ani1_onsite_global_dipole.pkl",
    spline="./data/opt/spline_1000ani1_onsite_global_dipole.pkl",
    dataset_dftb=dataset_dftb1,
    dataset_aims=dataset_aims1,
):
    """Plot loss function of two different ML method with different initial data."""
    target_aims = ["dipole", "charge", "hirshfeld_volume_ratio"]
    target_dftb = ["dipole", "charge"]
    # mprovements: 0.506583, 0.768370, 0.443744
    _, _, _, data_dftb = _load_ref(
        dataset_dftb, size_pred, target_dftb, test_ratio=test_ratio
    )
    _, _, geo_aims, data_aims = _load_ref(
        dataset_aims, size_pred, target_aims, test_ratio=test_ratio
    )
    comp_lr = pickle.load(open(comp_lr, "rb"))
    comp_gr = pickle.load(open(comp_gr, "rb"))
    spline = pickle.load(open(spline, "rb"))

    pred_lr = comp_lr.predict(geo_aims, ml_method="random_forest", feature_type="acsf")
    pred_gr = comp_gr.predict(geo_aims, ml_method="random_forest", feature_type="acsf")
    pred_spl = spline.predict(geo_aims, ml_method="random_forest", feature_type="acsf")

    ref_cpa = data_aims["hirshfeld_volume_ratio"]
    mio = Dftb2(geo_aims, shell_dict, path_to_skf="./slko/", skf_type="skf")
    mio()

    plt.plot([0, 1], [0, 1], "k")
    plt.plot(ref_cpa, mio.cpa, "b.", markersize=4, alpha=0.4)
    plt.plot(0, 0, "b.", label="mio-1-1")
    plt.plot(ref_cpa, pred_lr.cpa, "y.", markersize=4, alpha=0.4)
    plt.plot(0, 0, "y.", label="local basis")
    plt.plot(ref_cpa, pred_gr.cpa, "c.", markersize=4, alpha=0.4)
    plt.plot(0, 0, "c.", label="global basis")
    plt.plot(ref_cpa, pred_spl.cpa.detach(), "r.", markersize=4, alpha=0.4)
    plt.plot(0, 0, "r.", label="spline")
    plt.xlim(0.53, 0.67)
    plt.ylim(0.3, 1.25)
    plt.xlabel("DFT CPA of H")
    plt.ylabel("CPA predictions of H")
    plt.legend()
    plt.savefig('dip_predcpa.png', dpi=300)
    plt.show()

def other_pro_toc(
    comp_lr="./data/opt/vcr_1000ani1_onsite_local_dipole.pkl",
    comp_gr="./data/opt/vcr_1000ani1_onsite_global_dipole.pkl",
    spline="./data/opt/spline_1000ani1_onsite_global_dipole.pkl",
    dataset_dftb=dataset_dftb1,
    dataset_aims=dataset_aims1,
):
    """Plot loss function of two different ML method with different initial data."""
    target_aims = ["dipole", "charge", "hirshfeld_volume_ratio"]
    target_dftb = ["dipole", "charge"]
    # mprovements: 0.506583, 0.768370, 0.443744
    _, _, _, data_dftb = _load_ref(
        dataset_dftb, size_pred, target_dftb, test_ratio=test_ratio
    )
    _, _, geo_aims, data_aims = _load_ref(
        dataset_aims, size_pred, target_aims, test_ratio=test_ratio
    )
    comp_lr = pickle.load(open(comp_lr, "rb"))
    # comp_gr = pickle.load(open(comp_gr, "rb"))
    spline = pickle.load(open(spline, "rb"))

    pred_lr = comp_lr.predict(geo_aims, ml_method="random_forest", feature_type="acsf")
    # pred_gr = comp_gr.predict(geo_aims, ml_method="random_forest", feature_type="acsf")
    pred_spl = spline.predict(geo_aims, ml_method="random_forest", feature_type="acsf")

    ref_cpa = data_aims["hirshfeld_volume_ratio"]
    mio = Dftb2(geo_aims, shell_dict, path_to_skf="./slko/", skf_type="skf")
    mio()

    mask = pred_lr.geometry.atomic_numbers == 1
    n0, x0, _ = plt.hist(abs(ref_cpa - pred_lr.cpa)[mask], 15,
                         facecolor='r', alpha=0.2, density=True)
    bin_center0 = 0.5 * (x0[1:] + x0[:-1])
    print('n0', n0, n0.shape)
    print('x0', x0, x0.shape)

    plt.plot(bin_center0, n0, 'r', label='atomic fit')
    n1, x1, _ = plt.hist(
        abs(ref_cpa - pred_spl.cpa.detach())[mask], 30, facecolor='g',
        alpha=0.2, density=True)
    bin_center1 = 0.5 * (x1[1:] + x1[:-1])
    plt.plot(bin_center1, n1, 'g', label='diatomic fit')

    # plt.title('Error of property B', fontdict={'size': 20})
    plt.xticks(fontsize=12)
    plt.xlim(0, 0.6)
    plt.yticks([])
    # plt.legend(fontsize='xx-large')
    # plt.legend(loc='center', prop={'family': 'Arial', 'size': 12})
    plt.savefig("toc.png", dpi=300)
    plt.show()


def single_pro3(
    ani1_cha_compr="./data/opt/vcr_1000ani3_onsite_local_charge.pkl",
    ani1_dip_compr="./data/opt/vcr_1000ani3_onsite_local_dipole.pkl",
    ani1_cpa_compr="./data/opt/vcr_1000ani3_onsite_local_cpa.pkl",
    dataset_dftb=dataset_dftb3,
    dataset_aims=dataset_aims3,
):
    single_pro(
        ani1_cha_compr, ani1_dip_compr, ani1_cpa_compr, dataset_dftb, dataset_aims
    )


def single_pro(
    ani1_cha_compr="./data/opt/vcr_1000ani1_onsite_local_charge.pkl",
    ani1_dip_compr="./data/opt/vcr_1000ani1_onsite_local_dipole.pkl",
    ani1_cpa_compr="./data/opt/vcr_1000ani1_onsite_local_cpa.pkl",
    dataset_dftb=dataset_dftb1,
    dataset_aims=dataset_aims1,
):
    """Plot loss function of two different ML method with different initial data."""
    target_aims = ["dipole", "charge", "hirshfeld_volume_ratio"]
    target_dftb = ["dipole", "charge"]
    # mprovements: 0.506583, 0.768370, 0.443744
    _, _, _, data_dftb = _load_ref(
        dataset_dftb, size_pred, target_dftb, test_ratio=test_ratio
    )
    _, _, geo_aims, data_aims = _load_ref(
        dataset_aims, size_pred, target_aims, test_ratio=test_ratio
    )
    cha_compr = pickle.load(open(ani1_cha_compr, "rb"))
    cpa_compr = pickle.load(open(ani1_cpa_compr, "rb"))
    dip_compr = pickle.load(open(ani1_dip_compr, "rb"))

    predcha = cha_compr.predict(
        geo_aims, ml_method="random_forest", feature_type="acsf"
    )
    preddip = dip_compr.predict(
        geo_aims, ml_method="random_forest", feature_type="acsf"
    )
    predcpa = cpa_compr.predict(
        geo_aims, ml_method="random_forest", feature_type="acsf"
    )

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))  # sharex=True, sharey=True
    ldip = dip_compr.loss_list
    lcha = cha_compr.loss_list
    lcpa = cpa_compr.loss_list
    axs[0, 0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axs[0, 0].plot(
        np.linspace(1, len(ldip) - 1, len(ldip) - 1), ldip[1:], label="dipole"
    )
    axs[0, 0].plot(
        np.linspace(1, len(lcha) - 1, len(lcha) - 1), lcha[1:], label="charge"
    )
    axs[0, 0].plot(np.linspace(1, len(lcpa) - 1, len(lcpa) - 1), lcpa[1:], label="CPA")
    axs[0, 0].set_xlabel("steps", fontsize="large")
    axs[0, 0].set_ylabel("average loss", fontsize="large")
    axs[0, 0].legend(loc="best", fontsize="large")
    axs[0, 1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axs[0, 1].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    mask = data_aims["dipole"].ne(0)
    axs[0, 1].plot(np.linspace(-0.85, 0.7, 100), np.linspace(-0.85, 0.7, 100), "k")
    axs[0, 1].plot(
        data_aims["dipole"][mask], data_dftb["dipole"][mask], "b.",
        label="mio-1-1", markersize=3, alpha=0.3
    )
    axs[0, 1].plot(
        data_aims["dipole"][mask], preddip.dipole[mask], "r.",
        label="predictions", markersize=1.5, alpha=0.3
    )
    axs[0, 1].set_xlabel("reference dipole moments", fontsize="large")
    axs[0, 1].set_ylabel("DFTB dipole moments", fontsize="large")
    # axs[0, 1].legend(loc="best", fontsize="large")
    axs[0, 1].yaxis.set_label_position("right")
    axs[1, 0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axs[1, 0].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    print(
        1
        - abs(data_aims["dipole"][mask] - preddip.dipole[mask]).sum()
        / abs(data_aims["dipole"][mask] - data_dftb["dipole"][mask]).sum()
    )
    print(abs(data_aims["dipole"] - preddip.dipole).sum() / len(data_aims["dipole"]))
    print(
        abs(data_aims["dipole"] - data_dftb["dipole"]).sum() / len(data_aims["dipole"])
    )
    mask = data_aims["charge"].ne(0)
    axs[1, 0].plot(np.linspace(-0.8, 0.8, 10), np.linspace(-0.8, 0.8, 10), "k")
    axs[1, 0].plot(
        (data_aims["charge"] - predcha.qzero)[mask],
        (data_dftb["charge"] - predcha.qzero)[mask],
        "b.",
        label="mio-1-1",
        markersize=3,
        alpha=0.3
    )
    axs[1, 0].plot(
        (data_aims["charge"] - predcha.qzero)[mask],
        (predcha.charge - predcha.qzero)[mask],
        "r.",
        label="predictions",
        markersize=1.5,
        alpha=0.3
    )
    axs[1, 0].set_xlabel("reference charges", fontsize="large")
    axs[1, 0].set_ylabel("DFTB charges", fontsize="large")
    # axs[1, 0].legend(loc="best", fontsize="large")
    axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axs[1, 1].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    print(
        1
        - abs(data_aims["charge"][mask] - predcha.charge[mask]).sum()
        / abs(data_aims["charge"][mask] - data_dftb["charge"][mask]).sum()
    )
    print(abs(data_aims["charge"] - predcha.charge).sum() / len(data_aims["charge"]))
    print(
        abs(data_aims["charge"] - data_dftb["charge"]).sum() / len(data_aims["charge"])
    )
    data_aims["cpa"] = data_aims["hirshfeld_volume_ratio"]
    mask = data_aims["cpa"].ne(0)
    path_to_skf = "./slko/"
    dftb_mio = _cal_cpa(geo_aims, path_to_skf, skf_type="skf")
    axs[1, 1].plot(np.linspace(0.4, 1.0, 100), np.linspace(0.4, 1.0, 100), "k")
    axs[1, 1].plot(data_aims["cpa"][mask], dftb_mio.cpa[mask], "b.",
                   label="mio-1-1", markersize=3, alpha=0.3)
    axs[1, 1].plot(data_aims["cpa"][mask], predcpa.cpa[mask], "r.",
                   label="predictions", markersize=1.5, alpha=0.3)
    axs[1, 1].set_xlabel("reference CPA", fontsize="large")
    axs[1, 1].set_ylabel("DFTB CPA", fontsize="large")
    # axs[1, 1].legend(loc="best", fontsize="large")
    axs[1, 1].yaxis.set_label_position("right")
    print(
        1
        - abs(data_aims["cpa"][mask] - predcpa.cpa[mask]).sum()
        / abs(data_aims["cpa"][mask] - dftb_mio.cpa[mask]).sum()
    )
    print(abs(data_aims["cpa"] - predcpa.cpa).sum() / len(data_aims["cpa"]))
    print(abs(data_aims["cpa"] - dftb_mio.cpa).sum() / len(data_aims["cpa"]))
    plt.savefig("properties.png", dpi=300)
    plt.show()


def single_pro_mae(
    ani1_cha_compr="./data/opt/vcr_1000ani1_onsite_local_charge_data1.pkl",
    ani1_dip_compr="./data/opt/vcr_1000ani1_onsite_local_dipole_data1.pkl",
    ani1_cpa_compr="./data/opt/vcr_1000ani1_onsite_local_cpa_data1.pkl",
    dataset_dftb=dataset_dftb1,
    dataset_aims=dataset_aims1,
):
    # ->   dipole  charge  cpa
    # [0.1782, 0.1822, 0.1824, 0.1813, 0.1803, 0.1869, 0.1752, 0.1765, 0.1866] 0.006 0.006 dip low up
    # [0.1494, 0.1514, 0.1471, 0.1507, 0.1521, 0.1498, 0.1557, 0.1468, 0.1502] 0.004 0.005 cha low up
    # [0.2856, 0.2904, 0.2915, 0.2882, 0.2917, 0.2885, 0.2912, 0.2888, 0.2889] 0.004 0.002 cpa low up
    mio1 = [0.181, 0.1503, 0.289]
    # [0.0288, 0.0299, 0.0296] 0.0294 0.0007, 0.0005 dipole global basis1
    # [0.0732, 0.0764, 0.0756] 0.0751 0.0019 0.0013 charge global basis1
    # [0.1101, 0.1086, 0.1087] 0.0005 0.001 cpa global basis1
    r_glo1 = [0.0294, 0.0751, 0.109]
    # [0.01467, 0.01463, 0.01451] 0.0146 0.00009, 0.00006 dipole local basis1
    # [0.0241, 0.024, 0.0237] 0.239 0.00023, 0.00017 charge local basis1
    # [0.0658, 0.0686, 0.0676] 0.067, 0.0015, 0.0013 cpa local basis1
    r_loc1 = [0.0146, 0.0239, 0.067]
    # [0.0378, 0.037, 0.0356], 0.0012, 0.001 dipole spline1
    # [0.0474, 0.0461, 0.0467] 0.0467 0.0006 0.0007 charge spline1
    # [0.0869, 0.0958, 0.0977] 0.0935 0.0066 0.004 cpa spline1
    spline1 = [0.0368, 0.0467, 0.0935]
    # [0.2915, 0.2907, 0.2903, 0.4285, 0.4099, 0.4138, 0.4168, 0.4144, 0.4242] 0.007 0.12 dip low up
    # [0.3884, 0.3859, 0.3901, 0.4129, 0.4092, 0.414, 0.4194, 0.4138, 0.4101] 0.01 0.007 cha low up
    # [0.3419, 0.3423, 0.3417, 0.3417, 0.3405, 0.1, 0.3377, 0.1, 0.1] cpa low up
    mio3 = [0.295, 0.41, 0.34]
    # [0.2511, 0.2794, 0.2807, 0.2538, 0.2768, 0.2827, 0.2399] 0.026 0.016 dipole global basis3
    # [0.3405, 0.3412, 0.3600, 0.3598, 0.3507, 0.3105, 0.3573] 0.035 0.014 charge global basis3
    # [0.3218, 0.3234, 0.3237, 0.2877, 0.3195, 0.3218, 0.3178] 0.028 0.007 cpa global basis3
    r_glo3 = [0.2663, 0.3457, 0.3166]
    # [0.1037, 0.1047, 0.0991] 0.0034 0.0022 dipole local basis3
    # [0.0824, 0.0781, 0.0825] 0.0029 0.0015 charge local basis3
    # [0.10, 0.098, 0.098] 0.0006 0.001 cpa local basis3
    r_loc3 = [0.1025, 0.08, 0.099]
    # [0.2210, 0.2014, 0.2009, 0.2120, 0.2206, 0.2103, 0.2170] 0.011, 0.009 dipole spline3
    # [0.2365, 0.2380, 0.2251, 0.2373, 0.2455, 0.2426, 0.2627] 0.016 0.021 charge spline3
    # [0.2353, 0.2408, 0.2271, 0.2303, 0.2267, 0.2284, 0.2423]0.006 0.009 cpa splines
    spline3 = [0.2118, 0.2411, 0.2329]

    fig, axs = plt.subplots(3, 1, figsize=(7, 7))  # sharex=True, sharey=True
    ind = np.arange(2)  # -> xx
    width = 0.1

    # Dipole
    axs[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axs[0].bar(ind + 1 * width, (mio1[0], mio3[0]), width, fill=False,
               label="mio-1-1", hatch='--', edgecolor='red', alpha=0.8)
    axs[0].bar(ind + 2 * width, (r_glo1[0], r_glo3[0]), width, fill=False,
               label="global basis", hatch='xx', edgecolor='g', alpha=0.8)
    axs[0].bar(ind + 3 * width, (r_loc1[0], r_loc3[0]), width, fill=False,
               label="local basis", hatch='||', edgecolor='b', alpha=0.8)
    axs[0].bar(ind + 4 * width, (spline1[0], spline3[0]), width, fill=False,
               label="spline", hatch='++', edgecolor='c', alpha=0.8)
    axs[0].errorbar(  # error dipole mio1, mio3
        ind + 1 * width,
        (mio1[0], mio3[0]),
        yerr=[[0.006, 0.007], [0.006, 0.012]],
        fmt="k.",
    )
    axs[0].errorbar(  # error dipole_compr_global1, dipole_compr_global3
        ind + 2 * width,
        (r_glo1[0], r_glo3[0]),
        yerr=[[0.0007, 0.026], [0.0005, 0.016]],
        fmt="k.",
    )
    axs[0].errorbar(  # error dipole_compr_local1, dipole_compr_local3
        ind + 3 * width,
        (r_loc1[0], r_loc3[0]),
        yerr=[[0.00009, 0.0034], [0.00006, 0.0022]],
        fmt="k.",
    )
    axs[0].errorbar(  # error dipole_spline1, dipole_compr_spline3
        ind + 4 * width,
        (spline1[0], spline3[0]),
        yerr=[[0.0012, 0.011], [0.001, 0.009]],
        fmt="k.",
    )
    axs[0].set_xticks([])
    axs[0].set_ylabel("dipole MAE (au)", fontsize="large")
    axs[0].legend(loc="center", fontsize="large")

    # Charge
    axs[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axs[1].bar(ind + 1 * width, (mio1[1], mio3[1]), width, fill=False,
               label="mio-1-1", hatch='--', edgecolor='red', alpha=0.8)
    axs[1].bar(ind + 2 * width, (r_glo1[1], r_glo3[1]), width, fill=False,
               label="global basis", hatch='xx', edgecolor='g', alpha=0.8)
    axs[1].bar(ind + 3 * width, (r_loc1[1], r_loc3[1]), width, fill=False,
               label="local basis", hatch='||', edgecolor='b', alpha=0.8)
    axs[1].bar(ind + 4 * width, (spline1[1], spline3[1]), width, fill=False,
               label="spline", hatch='++', edgecolor='c', alpha=0.8)
    axs[1].set_xticks([])
    axs[1].errorbar(  # error dipole mio1, mio3
        ind + 1 * width,
        (mio1[1], mio3[1]),
        yerr=[[0.004, 0.01], [0.005, 0.007]],
        fmt="k.",
    )
    axs[1].errorbar(  # error charge_compr_global1, charge_compr_global3
        ind + 2 * width,
        (r_glo1[1], r_glo3[1]),
        yerr=[[0.0019, 0.035], [0.0013, 0.014]],
        fmt="k.",
    )
    axs[1].errorbar(  # error charge_compr_local1, charge_compr_local3
        ind + 3 * width,
        (r_loc1[1], r_loc3[1]),
        yerr=[[0.00023, 0.0029], [0.00017, 0.0015]],
        fmt="k.",
    )
    axs[1].errorbar(  # error charge_spline1, charge_spline3
        ind + 4 * width,
        (spline1[1], spline3[1]),
        yerr=[[0.0006, 0.016], [0.0007, 0.021]],
        fmt="k.",
    )
    axs[1].legend(loc="center", fontsize="large")
    axs[1].set_ylabel("charge MAE (au)", fontsize="large")

    axs[2].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axs[2].bar(ind + 1 * width, (mio1[2], mio3[2]), width, fill=False,
               label="mio-1-1", hatch='--', edgecolor='red', alpha=0.8)
    axs[2].bar(ind + 2 * width, (r_glo1[2], r_glo3[2]), width, fill=False,
               label="global basis", hatch='xx', edgecolor='g', alpha=0.8)
    axs[2].bar(ind + 3 * width, (r_loc1[2], r_loc3[2]), width, fill=False,
               label="local basis", hatch='||', edgecolor='b', alpha=0.8)
    axs[2].bar(ind + 4 * width, (spline1[2], spline3[2]), width, fill=False,
               label="spline", hatch='++', edgecolor='red', alpha=0.8)
    axs[2].set_ylabel("CPA MAE (au)", fontsize="large")
    axs[2].legend(loc="best", fontsize="large")
    axs[2].set_xticks([0.25, 1.25], [r"ANI-1$_1$", r"ANI-1$_3$"], fontsize="x-large")
    axs[2].errorbar(  # error dipole mio1, mio3
        ind + 1 * width,
        (mio1[2], mio3[2]),
        yerr=[[0.004, 0.01], [0.002, 0.007]],
        fmt="k.",
    )
    axs[2].errorbar(  # error cpa_compr_global1, cpa_compr_global3
        ind + 2 * width,
        (r_glo1[2], r_glo3[2]),
        yerr=[[0.0005, 0.028], [0.001, 0.007]],
        fmt="k.",
    )
    axs[2].errorbar(  # error cpa_compr_local1, cpa_compr_local3
        ind + 3 * width,
        (r_loc1[2], r_loc3[2]),
        yerr=[[0.0015, 0.0006], [0.0013, 0.001]],
        fmt="k.",
    )
    axs[2].errorbar(  # error cpa_spline1, cpa_spline3
        ind + 4 * width,
        (spline1[2], spline3[2]),
        yerr=[[0.0066, 0.006], [0.0042, 0.009]],
        fmt="k.",
    )
    plt.savefig('single_pro_mae.png', dpi=300)
    plt.show()


def mul_pro(dataset_dftb=dataset_dftb1, dataset_aims=dataset_aims1):
    """Plot loss function of two different ML method with different initial data."""
    target_aims = ["dipole", "charge", "hirshfeld_volume_ratio"]
    target_dftb = ["dipole", "charge"]
    # mprovements: 0.506583, 0.768370, 0.443744
    _, _, _, data_dftb = _load_ref(
        dataset_dftb, size_pred, target_dftb, test_ratio=test_ratio
    )
    _, _, geo_aims, data_aims = _load_ref(
        dataset_aims, size_pred, target_aims, test_ratio=test_ratio
    )

    fig, axs = plt.subplots(2, 1, figsize=(7, 7))  # sharex=True, sharey=True
    width = 1
    dip1_mio = [0.1354, 0.1367, 0.1373, 0.1336, 0.1356, 0.1322]
    cha1_mio = [0.1409, 0.1494, 0.1460, 0.1468, 0.1502, 0.1444]
    cpa1_mio = [0.2941, 0.2905, 0.2920, 0.2931, 0.2916, 0.2871]
    dip1_mio_av = sum(dip1_mio) / len(dip1_mio)
    cha1_mio_av = sum(cha1_mio) / len(cha1_mio)
    cpa1_mio_av = sum(cpa1_mio) / len(cpa1_mio)
    dip1_mio_err = [[dip1_mio_av - min(dip1_mio)], [max(dip1_mio) - dip1_mio_av]]
    cha1_mio_err = [[cha1_mio_av - min(cha1_mio)], [max(cha1_mio) - cha1_mio_av]]
    cpa1_mio_err = [[cpa1_mio_av - min(cpa1_mio)], [max(cpa1_mio) - cpa1_mio_av]]
    dip1_dipcha = [0.0598, 0.0622, 0.0600]
    cha1_dipcha = [0.1139, 0.1162, 0.1190]
    dip1_dipcpa = [0.0255, 0.0244, 0.0260]
    cpa1_dipcpa = [0.1042, 0.1035, 0.1037]
    cha1_chacpa = [0.0466, 0.0481, 0.0479]
    cpa1_chacpa = [0.1155, 0.1168, 0.1114]
    dip1_av = [sum(dip1_dipcha) / len(dip1_dipcha), sum(dip1_dipcpa) / len(dip1_dipcpa)]
    cha1_av = [sum(cha1_dipcha) / len(cha1_dipcha), sum(cha1_chacpa) / len(cha1_chacpa)]
    cpa1_av = [sum(cpa1_dipcpa) / len(cpa1_dipcpa), sum(cpa1_chacpa) / len(cpa1_chacpa)]
    dip1_err = [
        [dip1_av[0] - min(dip1_dipcha), dip1_av[1] - min(dip1_dipcpa)],
        [max(dip1_dipcha) - dip1_av[0], max(dip1_dipcpa) - dip1_av[1]],
    ]
    cha1_err = [
        [cha1_av[0] - min(cha1_dipcha), cha1_av[1] - min(cha1_chacpa)],
        [max(cha1_dipcha) - cha1_av[0], max(cha1_chacpa) - cha1_av[1]],
    ]
    cpa1_err = [
        [cpa1_av[0] - min(cpa1_dipcpa), cpa1_av[1] - min(cpa1_chacpa)],
        [max(cpa1_dipcpa) - cpa1_av[0], max(cpa1_chacpa) - cpa1_av[1]],
    ]

    dip3_mio = [0.2861, 0.2890, 0.2902, 0.2910, 0.2952, 0.2858]
    cha3_mio = [0.3876, 0.3813, 0.3902, 0.3795, 0.3886, 0.3833]
    cpa3_mio = [0.3398, 0.3410, 0.3440, 0.3397, 0.3400, 0.3419]
    dip3_mio_av = sum(dip3_mio) / len(dip3_mio)
    cha3_mio_av = sum(cha3_mio) / len(cha3_mio)
    cpa3_mio_av = sum(cpa3_mio) / len(cpa3_mio)
    dip3_mio_err = [[dip3_mio_av - min(dip3_mio)], [max(dip3_mio) - dip3_mio_av]]
    cha3_mio_err = [[cha3_mio_av - min(cha3_mio)], [max(cha3_mio) - cha3_mio_av]]
    cpa3_mio_err = [[cpa3_mio_av - min(cpa3_mio)], [max(cpa3_mio) - cpa3_mio_av]]
    dip3_dipcha = [0.0869, 0.0922, 0.0867]
    cha3_dipcha = [0.1383, 0.1382, 0.1333]
    dip3_dipcpa = [0.0934, 0.0930, 0.0953]
    cpa3_dipcpa = [0.1930, 0.1753, 0.1744]
    cha3_chacpa = [0.1123, 0.1105, 0.1117]
    cpa3_chacpa = [0.1608, 0.1630, 0.1634]
    dip3_av = [sum(dip3_dipcha) / len(dip3_dipcha), sum(dip3_dipcpa) / len(dip3_dipcpa)]
    cha3_av = [sum(cha3_dipcha) / len(cha3_dipcha), sum(cha3_chacpa) / len(cha3_chacpa)]
    cpa3_av = [sum(cpa3_dipcpa) / len(cpa3_dipcpa), sum(cpa3_chacpa) / len(cpa3_chacpa)]
    dip3_err = [
        [dip3_av[0] - min(dip3_dipcha), dip3_av[1] - min(dip3_dipcpa)],
        [max(dip3_dipcha) - dip3_av[0], max(dip3_dipcpa) - dip3_av[1]],
    ]
    cha3_err = [
        [cha3_av[0] - min(cha3_dipcha), cha3_av[1] - min(cha3_chacpa)],
        [max(cha3_dipcha) - cha3_av[0], max(cha3_chacpa) - cha3_av[1]],
    ]
    cpa3_err = [
        [cpa3_av[0] - min(cpa3_dipcpa), cpa3_av[1] - min(cpa3_chacpa)],
        [max(cpa3_dipcpa) - cpa3_av[0], max(cpa3_chacpa) - cpa3_av[1]],
    ]

    axs[1].xaxis.tick_top()
    # Dipole
    axs[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axs[0].bar([0], dip1_mio_av, width, color="r", label="dipole",
               fill=False, hatch='--', edgecolor='red', alpha=0.8)
    axs[0].bar([1], cha1_mio_av, width, color="orange", label="charge",
               fill=False, hatch='||', edgecolor='b', alpha=0.8)
    axs[0].bar([2], cpa1_mio_av, width, color="c", label="CPA",
               fill=False, hatch='xx', edgecolor='c', alpha=0.8)
    axs[0].errorbar([0], dip1_mio_av, yerr=dip1_mio_err, fmt="k.")
    axs[0].errorbar([1], cha1_mio_av, yerr=cha1_mio_err, fmt="k.")
    axs[0].errorbar([2], cpa1_mio_av, yerr=cpa1_mio_err, fmt="k.")
    axs[0].bar([4, 10], dip1_av, width, color="r",
               fill=False, hatch='--', edgecolor='red', alpha=0.8)
    axs[0].bar([5, 7], cha1_av, width, color="orange",
               fill=False, hatch='||', edgecolor='b', alpha=0.8)
    axs[0].bar([8, 11], cpa1_av, width, color="c",
               fill=False, hatch='xx', edgecolor='c', alpha=0.8)
    axs[0].errorbar([4, 10], dip1_av, yerr=dip1_err, fmt="k.",)
    axs[0].errorbar([5, 7], cha1_av, yerr=cha1_err, fmt="k.")
    axs[0].errorbar([8, 11], cpa1_av, yerr=cpa1_err, fmt="k.",)
    axs[0].set_xticks([])
    axs[0].text(5, 0.25, r"ANI-1$_1$", fontsize="x-large")
    axs[0].set_ylabel("MAE (au)", fontsize="large")
    axs[0].legend(loc="best", fontsize="large")

    axs[1].set_ylabel("MAE (au)", fontsize="large")
    axs[1].bar([0], dip3_mio_av, width, color="r", label="dipole",
               fill=False, hatch='--', edgecolor='red', alpha=0.8)
    axs[1].bar([1], cha3_mio_av, width, color="orange", label="charge",
               fill=False, hatch='||', edgecolor='b', alpha=0.8)
    axs[1].bar([2], cpa3_mio_av, width, color="c", label="CPA",
               fill=False, hatch='xx', edgecolor='c', alpha=0.8)
    axs[1].errorbar([0], dip3_mio_av, yerr=dip3_mio_err, fmt="k.")
    axs[1].errorbar([1], cha3_mio_av, yerr=cha3_mio_err, fmt="k.")
    axs[1].errorbar([2], cpa3_mio_av, yerr=cpa3_mio_err, fmt="k.")
    axs[1].bar([4, 10], dip3_av, width, color="r",
               fill=False, hatch='--', edgecolor='red', alpha=0.8)
    axs[1].bar([5, 7], cha3_av, width, color="orange",
               fill=False, hatch='||', edgecolor='b', alpha=0.8)
    axs[1].bar([8, 11], cpa3_av, width, color="c",
               fill=False, hatch='xx', edgecolor='c', alpha=0.8)
    axs[1].errorbar([4, 10], dip3_av, yerr=dip3_err, fmt="k.")
    axs[1].errorbar([5, 7], cha3_av, yerr=cha3_err, fmt="k.")
    axs[1].errorbar([8, 11], cpa3_av, yerr=cpa3_err, fmt="k.")
    axs[1].get_shared_x_axes().join(axs[0], axs[1])
    axs[1].text(5, 0.32, r"ANI-1$_3$", fontsize="x-large")
    axs[1].set_xticks(
        [1, 4.5, 7.5, 10.5],
        ("MIO", "dipole-charge", "charge-CPA", "dipole-CPA"),
    )
    plt.legend(loc="best", fontsize="large")
    plt.savefig("multipro.png", dpi=300)
    plt.show()


def trans_dataset():
    """Test optimized results."""
    ani_pred = 5  # decide which file to load
    size_pred = 400
    test_ratio = 1  # Select from all dataset, could be trained data
    pickle_file = "./data/opt/vcr_1000ani3_onsite_local_charge.pkl"
    from train_npe import _load_ref, _plot, _cal_cpa

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

    # load optimized object
    loaded_model = pickle.load(open(pickle_file, "rb"))

    plt.plot(loaded_model.reference["charge"], loaded_model.reference["charge"], "k")
    plt.plot(
        loaded_model.reference["charge"],
        getattr(loaded_model.dftb, "charge").detach(),
        "rx",
    )
    plt.show()

    _, data_dftb = _load_ref(
        dataset_dftb_pred, size_pred, ["charge", "dipole"], test_ratio=test_ratio
    )
    geo_aims, data_aims = _load_ref(
        dataset_aims_pred,
        size_pred,
        ["charge", "dipole", "hirshfeld_volume_ratio"],
        test_ratio=test_ratio,
    )
    dftb = loaded_model.predict(
        geo_aims, ml_method="random_forest", feature_type="acsf"
    )

    dftb_mio = _cal_cpa(geo_aims, path="./slko/mio_new.hdf")
    _plot(data_aims["hirshfeld_volume_ratio"], dftb.cpa, dftb_mio.cpa, "cpa")

    _plot(data_aims["charge"], dftb.charge, dftb_mio.charge, "charge")

    fig, axs = plt.subplots(1, 2, figsize=(7, 7))  # sharex=True, sharey=True
    # Dipole
    axs[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axs[0].plot([-0.5, 0, 0.5], [-0.5, 0, 0.5], "k")
    axs[0].plot(data_aims["charge"] - dftb.qzero, dftb_mio.charge - dftb.qzero,
                "b.", markersize=4, alpha=0.3)
    axs[0].plot(data_aims["charge"] - dftb.qzero, dftb.charge - dftb.qzero,
                "r.", markersize=2, alpha=0.5)
    axs[0].plot(0, 0, "b.", label="mio-1-1")
    axs[0].plot(0, 0, "rx", label="predictions")
    axs[0].legend(fontsize="large")
    axs[0].set_xlabel("DFT charge", fontsize="x-large")
    axs[0].set_ylabel("DFTB charge", fontsize="x-large")
    axs[1].bar(0, 0.60, width=0.5, fill=False, hatch='++', edgecolor='g',alpha=0.8)
    axs[1].bar(1, 0.26, width=0.5, fill=False, hatch='xx', edgecolor='c', alpha=0.8)
    axs[1].set_ylabel("charge MAE (au)", fontsize="x-large")
    axs[1].set_xticks([0, 1], ["mio-1-1", "predictions"], fontsize="x-large")
    axs[1].yaxis.set_label_position("right")
    plt.savefig('transcha.png', dpi=300)
    plt.show()


def frame_type():
    """Train with size 1400, test with 400."""
    compr_err1 = [0.1365, 0.0128, 0.039, 0.035]
    # MIO, local prediction, global prediction, spline
    compr_err3 = [0.4139, 0.071, 0.28, 0.20]

    ind = np.arange(2)  # -> xx
    width = 0.15  # Width of a bar

    plt.bar(
        ind + 0 * width,
        (compr_err1[0], compr_err3[0]),
        width,
        label="previous parameters",
    )
    plt.bar(
        ind + 1 * width,
        (compr_err1[1], compr_err3[1]),
        width,
        label="local predictions",
    )
    plt.bar(
        ind + 2 * width,
        (compr_err1[2], compr_err3[2]),
        width,
        label="global predictions",
        color="seagreen",
    )
    plt.bar(
        ind + 3 * width,
        (compr_err1[3], compr_err3[3]),
        width,
        label="spline model predictions",
        color="indianred",
    )
    # plt.ylim(0, 1)
    # plt.ylabel('Improvements of dipole moments', fontsize='large')
    plt.ylabel("Mean absolute error (au)", fontsize="large")
    # xticks()
    plt.xticks(ind + 2.5 * width / 2, (r"$ANI-1_1$", r"$ANI-1_3$"), fontsize="large")
    # Finding the best position for legends and putting it
    plt.legend(loc="upper left", fontsize="large")
    plt.savefig("MLmethod.png", dpi=300)
    plt.show()


def ml_method():
    """Train with size 800, test with 400."""
    mio1 = [0.1369, 0.1365, 0.1333, 0.1363, 0.1331, 0.1362]
    mio3 = [0.2908, 0.2918, 0.2818, 0.2852, 0.2873, 0.2918]
    train1 = [0.0069, 0.0068, 0.0066, 0.0069, 0.0068, 0.0066]
    train3 = [0.0329, 0.0325, 0.0322, 0.0328, 0.0325, 0.0322]
    nn1 = [0.0501, 0.0489, 0.0499]
    nn3 = [0.2140, 0.2139, 0.1991]
    rf1 = [0.0133, 0.0136, 0.0137]
    rf3 = [0.0939, 0.0967, 0.1009]
    mio_av = [sum(mio1) / len(mio1), sum(mio3) / len(mio3)]
    train_av = [sum(train1) / len(train1), sum(train3) / len(train3)]
    nn_av = [sum(nn1) / len(nn1), sum(nn3) / len(nn3)]
    rf_av = [sum(rf1) / len(rf1), sum(rf3) / len(rf3)]
    mio_err = [
        [mio_av[0] - min(mio1), mio_av[1] - min(mio3)],
        [max(mio1) - mio_av[0], max(mio3) - mio_av[1]],
    ]
    train_err = [
        [train_av[0] - min(train1), train_av[1] - min(train3)],
        [max(train1) - train_av[0], max(train3) - train_av[1]],
    ]
    nn_err = [
        [nn_av[0] - min(nn1), nn_av[1] - min(nn3)],
        [max(nn1) - nn_av[0], max(nn3) - nn_av[1]],
    ]
    rf_err = [
        [rf_av[0] - min(rf1), rf_av[1] - min(rf3)],
        [max(rf1) - rf_av[0], max(rf3) - rf_av[1]],
    ]

    ind = np.arange(2)  # -> xx
    width = 0.15  # Width of a bar

    labels = ["MIO", "training", "ACSF + RF predictions", "ACSF + NN predictions"]
    plt.bar(ind + 0 * width, mio_av, width, label=labels[0], fill=False,
            hatch='--', edgecolor='g', alpha=0.8)
    plt.errorbar(ind + 0 * width, mio_av, mio_err, fmt="k.")
    plt.bar(ind + 1 * width, train_av, width, label=labels[1], fill=False,
            hatch='xx', edgecolor='r', alpha=0.8)
    plt.errorbar(ind + 1 * width, train_av, train_err, fmt="k.")
    plt.bar(ind + 2 * width, rf_av, width, label=labels[2], fill=False,
            hatch='||', edgecolor='b', alpha=0.8)
    plt.errorbar(ind + 2 * width, rf_av, rf_err, fmt="k.")
    plt.bar(ind + 3 * width, nn_av, width, label=labels[3], fill=False,
            hatch='++', edgecolor='c', alpha=0.8)
    plt.errorbar(ind + 3 * width, nn_av, nn_err, fmt="k.")
    plt.ylabel("MAE of dipole (au)", fontsize="large")
    # xticks()
    plt.xticks(ind + 2.9 * width / 2, (r"$ANI-1_1$", r"$ANI-1_3$"), fontsize="large")
    # Finding the best position for legends and putting it
    plt.legend(loc="upper left", fontsize="large")
    plt.savefig("MLmethod.png", dpi=300)
    plt.show()


def test_compr():
    from ase.build import molecule
    path_to_skf = "./vcr.h5"
    h2o = molecule("H2O")
    geometry = Geometry.from_ase_atoms([h2o])
    grids = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0])
    multi_varible = torch.tensor([[3.0, 3.0, 3.0]])
    dftb2 = Dftb2(
        geometry,
        shell_dict=shell_dict,
        path_to_skf=path_to_skf,
        skf_type="h5",
        basis_type="vcr",
        interpolation="BicubInterp",
        grids=grids,
        multi_varible=multi_varible,
        repulsive=False
    )
    dftb2()

    path_to_skf = "./tests/unittests/data/slko/vcr.h5"
    h2o = molecule("H2O")
    geometry = Geometry.from_ase_atoms([h2o])
    grids = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0])
    dftb2 = Dftb2(
        geometry,
        shell_dict=shell_dict,
        path_to_skf=path_to_skf,
        skf_type="h5",
        basis_type="vcr",
        interpolation="BicubInterp",
        grids=grids,
        multi_varible=multi_varible,
    )
    dftb2()


def TOC(
    # ani1_dip_compr="./data/opt/vcr_1000ani1_onsite_local_dipole.pkl",
    ani1_dip_compr="./vcr_100ani1_onsite_local_dipole.pkl",
    dataset_dftb=dataset_dftb1,
    dataset_aims=dataset_aims1,
):
    """Plot loss function of two different ML method with different initial data."""
    target_aims = ["dipole", "charge", "hirshfeld_volume_ratio"]
    target_dftb = ["dipole", "charge"]
    # mprovements: 0.506583, 0.768370, 0.443744
    _, _, _, data_dftb = _load_ref(
        dataset_dftb, size_pred, target_dftb, test_ratio=test_ratio
    )
    _, _, geo_aims, data_aims = _load_ref(
        dataset_aims, size_pred, target_aims, test_ratio=test_ratio
    )
    dip_compr = pickle.load(open(ani1_dip_compr, "rb"))

    preddip = dip_compr.predict(
        geo_aims, ml_method="random_forest", feature_type="acsf"
    )

    ldip = dip_compr.loss_list

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

    # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xticks([])
    ax.set_yticks([])
    mask = data_aims["dipole"].ne(0)
    ax.plot(np.linspace(-0.85, 0.7, 100), np.linspace(-0.85, 0.7, 100), "k")
    ax.plot(
        data_aims["dipole"][mask], preddip.dipole[mask], "rx", label="DFTB-ML"
    )
    ax.plot(
        data_aims["dipole"][mask], data_dftb["dipole"][mask], "bo", label="standard DFTB "
    )
    ax.set_xlabel("DFT dipole", fontsize="large")
    ax.set_ylabel("DFTB dipole", fontsize="large")
    ax.legend(loc="best", fontsize="large")
    # ax.yaxis.set_label_position("right")

    plt.savefig("TOC.png", dpi=300)
    plt.show()


def test_random_data(
    # ani1_dip_compr="./data/opt/vcr_1000ani1_onsite_local_dipole.pkl",
    ani1_dip_compr="./data/opt/vcr_1000ani1_onsite_local_dipole.pkl",
    ani1_dip_compr2="./data/opt/vcr_1000ani1_onsite_local_dipole2.pkl",
    dataset_dftb=dataset_dftb1,
    dataset_aims=dataset_aims1,
):
    """Plot loss function of two different ML method with different initial data."""
    target_aims = ["dipole", "charge", "hirshfeld_volume_ratio"]
    target_dftb = ["dipole", "charge"]
    # mprovements: 0.506583, 0.768370, 0.443744
    _, _, _, data_dftb = _load_ref(
        dataset_dftb, size_pred, target_dftb, test_ratio=test_ratio
    )
    _, _, geo_aims, data_aims = _load_ref(
        dataset_aims, size_pred, target_aims, test_ratio=test_ratio
    )
    dip_compr1 = pickle.load(open(ani1_dip_compr, "rb"))
    dip_compr2 = pickle.load(open(ani1_dip_compr2, "rb"))

    preddip1 = dip_compr1.predict(
        geo_aims, ml_method="random_forest", feature_type="acsf"
    )
    preddip2 = dip_compr2.predict(
        geo_aims, ml_method="random_forest", feature_type="acsf"
    )

    mask = data_aims["dipole"].ne(0)
    error = preddip1.dipole[mask] - preddip2.dipole[mask]
    plt.hist(error[error.ne(0)], 50, density=True, facecolor='g', alpha=0.75)
    plt.ylabel('counts')
    plt.xlabel('dipole error')
    plt.savefig("TOC.png", dpi=300)
    plt.show()


# def pred_large_mol_cha(
#     ani1_cha_compr="./data/opt/vcr_1000ani1_onsite_local_dipole_data1.pkl",
#     ani1_cha_int="./data/opt/spline_1000ani1_onsite_global_dipole_data1.pkl",
#     ani3_cha_compr="./data/opt/vcr_1000ani3_onsite_local_charge_data1.pkl",
#     ani3_cha_int="./data/opt/spline_1000ani3_onsite_global_dipole_data1.pkl",
#     dataset_dftb=dataset_dftb3,
#     dataset_aims=dataset_aims3):
#     """Use trained model to predict organic molecule."""
#     path_to_skf = "../unittests/data/slko/mio/"
#     # organic moleculke: 1, acetic acid, 2. Glycin
#     geo_mol = Geometry(
#         atomic_numbers=torch.tensor(
#             [[8, 8, 6, 6, 1, 1, 1, 1, 0, 0, 0, 0],
#              [6, 6, 8, 8, 7, 1, 1, 1, 1, 1, 0, 0],
#              [8, 6, 1, 6, 1, 1, 1, 0, 0, 0, 0, 0],
#              [8, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]]),
#         positions=torch.tensor(
#          [[[-0.789989, -1.039953, -0.000039], [-0.635145, 1.208040, 0.000016],
#            [1.397576, -0.119879, -0.000058], [-0.088869, 0.125419, -0.000137],
#            [1.680531, -0.699298, 0.885534], [1.679818, -0.708633, -0.879622],
#            [1.925614, 0.834244, -0.004979], [-1.737135, -0.804249, 0.000418],
#            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
#           [[0.539406, -0.102463, -0.017685], [-0.731936, 0.711159, -0.137239],
#            [1.634766, 0.688507, 0.085730], [0.600633, -1.312652, -0.053054],
#            [-1.885996, -0.103617, 0.216045], [-0.735353, 1.106336, -1.170608],
#            [-0.659850, 1.578195, 0.528483], [2.415042, 0.102412, 0.106245],
#            [-1.798528, -1.035448, -0.184631], [-2.747356, 0.314812, -0.123664],
#            [0, 0, 0], [0, 0, 0]],
#           [[1.218055, 0.36124, 0.0], [0.0, 0.464133, 0.0],
#            [-0.477241, 1.465295, 0.0], [-0.948102, -0.700138, 0.0],
#            [-0.385946, -1.634236, 0.0], [-1.596321, -0.652475, 0.880946],
#            [-1.596321, -0.652475, -0.880946],
#            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],],
#           [[0.006429, -0.712741, 0.0], [0.0, 0.705845, 0.0],
#            [1.324518, -1.226029, 0.0], [-1.442169, 1.160325, 0.0],
#            [0.530962, 1.086484, 0.886881], [0.530962, 1.086484, -0.886881],
#            [1.241648, -2.313325, 0.0], [1.881329, -0.905925, -0.89171],
#            [1.881329, -0.905925, 0.89171], [-1.954863, 0.780605, -0.885855],
#            [-1.954863, 0.780605, 0.885855], [-1.502025, 2.252083, 0.0],]]),
#         units="angstrom")
#     ref_charge = torch.tensor(
#         [[6.290607, 6.327366, 4.171931,3.728239,0.921185,0.921208,0.909664,
#           0.729799,0,0,0,0,],
#          [3.719969,4.063653,6.284300,6.333901,5.394457,0.928597,0.903539,
#           0.728024,0.816907,0.826653,0,0,],
#          [6.282040,3.868290,0.933666,4.148259,0.918044,0.924851,0.924851,
#           0.0,0.0,0,0,0,],
#          [6.200489,4.034680,4.124958,4.165459,0.942072,0.942072,0.902250,
#           0.936509,0.936509,0.934970,0.934970,0.945061,],])
#     cha_compr = pickle.load(open(ani1_cha_compr, "rb"))
#     cha_int = pickle.load(open(ani1_cha_int, "rb"))
#     cha_compr3 = pickle.load(open(ani3_cha_compr, "rb"))
#     cha_int3 = pickle.load(open(ani1_cha_int, "rb"))
#     dftb2 = Dftb2(geometry=geo_mol, shell_dict=shell_dict, path_to_skf=path_to_skf, skf_type="skf")
#     dftb2()
#     cha_compr3.params["ml_method"] = "random_forest"
#     # ACSF+RF
#     predchaar = cha_compr.predict(geo_mol, ml_method="random_forest")
#     predcha3ar = cha_compr3.predict(geo_mol, ml_method="random_forest")
#     print("predchaar.charge", predcha3ar.charge)
#     mask = ref_charge.ne(0)
#     qzero = predchaar.qzero
#     plt.plot((ref_charge - qzero)[mask], (ref_charge - qzero)[mask], color="k")
#     plt.plot((ref_charge - qzero)[mask], (dftb2.charge - qzero)[mask], "oc", label="MIO")
#     plt.plot((ref_charge - qzero)[mask], (predcha3ar.charge - qzero)[mask],
#         ".r", label="WCR ANI-1-3-AR",)
#     plt.ylabel("reference charge")
#     plt.xlabel("calculated (predicted) charge")
#     plt.legend()
#     print("mio", abs(dftb2.charge - ref_charge).sum(-1),
#         "ani_1_ra",abs(predchaar.charge - ref_charge).sum(-1),
#         "ani_3_ra", abs(predcha3ar.charge - ref_charge).sum(-1))


def _load_ref(dataset, size, properties, test_ratio, units="angstrom", **kwargs):
    """Helper function to load dataset, return `Geometry` object, data."""
    numbers, positions, data = LoadHdf.load_reference(
        dataset, size, properties, test_ratio=test_ratio, version=version
    )
    cell = kwargs.get("cell", None)
    geo = Geometry(numbers, positions, units=units, cell=cell)

    return numbers, positions, geo, data


def _cal_cpa(geometry, path_to_skf, skf_type):
    shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
    dftb = Dftb2(geometry, shell_dict, path_to_skf, skf_type=skf_type)
    dftb()
    return dftb


if __name__ == "__main__":
    # pred_dip_size()
    # other_pro()
    # single_pro()
    # single_pro3()
    # ml_method()
    # single_pro_mae()
    # mul_pro()
    # test_compr()
    # trans_dataset()
    other_pro_toc()
    TOC()
    test_random_data()
