import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.common.modules.common import MLP, PatchEmbeddings
from src.common.utils import get_modality_combinations


CLINICAL_COLUMNS = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "PTETHCAT",
    "PTRACCAT",
    "PTMARRY",
    "APOE4",
    "CDRSB",
    "ADAS11",
    "ADAS13",
    "ADASQ4",
    "MMSE",
    "RAVLT_immediate",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
    "LDELTOTAL",
    "DIGITSCOR",
    "TRABSCOR",
    "FAQ",
    "MOCA",
    "EcogPtMem",
    "EcogPtLang",
    "EcogPtVisspat",
    "EcogPtPlan",
    "EcogPtOrgan",
    "EcogPtDivatt",
    "EcogPtTotal",
    "EcogSPMem",
    "EcogSPLang",
    "EcogSPVisspat",
    "EcogSPPlan",
    "EcogSPOrgan",
    "EcogSPDivatt",
    "EcogSPTotal",
]

BIOSPECIMEN_COLUMNS = [
    "FDG",
    "PIB",
    "AV45",
    "FBB",
    "ABETA",
    "TAU",
    "PTAU",
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "ICV",
]

LABEL_MAP = {"CN": 0, "MCI": 1, "Dementia": 2}


def _device_from_args(args):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.device}")
    return torch.device("cpu")


def _prepare_features(df, columns):
    existing_columns = [col for col in columns if col in df.columns]
    if not existing_columns:
        raise ValueError(f"No usable columns found among: {columns}")

    features = df[existing_columns].copy()
    numeric_features = features.apply(pd.to_numeric, errors="ignore")
    features = pd.get_dummies(numeric_features, dummy_na=True)
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

    arr = features.to_numpy(dtype=np.float32)
    if arr.shape[1] > 0:
        arr = MinMaxScaler(feature_range=(-1, 1)).fit_transform(arr)
    return arr.astype(np.float32)


def _build_encoder(input_dim, args, device):
    if args.patch:
        return PatchEmbeddings(input_dim, args.num_patches, args.hidden_dim).to(device)
    return MLP(input_dim, args.hidden_dim, args.hidden_dim, args.num_layers_enc).to(
        device
    )


def load_and_preprocess_data_adni_demo(args):
    csv_path = "data/adni_demo/ADNIMERGE_06May2026.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ADNI demo CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["VISCODE"].eq("bl")].copy()
    df = df[df["DX"].isin(LABEL_MAP)].copy()
    df = df.drop_duplicates(subset="PTID", keep="first").reset_index(drop=True)

    if getattr(args, "debug", False):
        df = df.groupby("DX", group_keys=False).head(40).reset_index(drop=True)

    labels = df["DX"].map(LABEL_MAP).to_numpy(dtype=np.int64)
    n_labels = len(LABEL_MAP)
    all_ids = np.arange(len(df))

    train_ids, holdout_ids = train_test_split(
        all_ids,
        test_size=0.3,
        random_state=args.seed,
        stratify=labels,
    )
    valid_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=2 / 3,
        random_state=args.seed,
        stratify=labels[holdout_ids],
    )

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}
    common_idx_list = []
    observed_idx_arr = np.zeros((len(df), len(args.modality)), dtype=bool)
    modality_combinations = [""] * len(df)
    device = _device_from_args(args)

    def add_modality(char, name, columns):
        arr = _prepare_features(df, columns)
        data_dict[name] = arr
        input_dims[name] = arr.shape[1]
        encoder_dict[name] = _build_encoder(arr.shape[1], args, device)
        filtered_idx = set(range(len(df)))
        common_idx_list.append(filtered_idx)
        modality_position = args.modality.upper().index(char)
        observed_idx_arr[:, modality_position] = True
        for idx in filtered_idx:
            modality_combinations[idx] += char

    if "C" in args.modality.upper():
        add_modality("C", "clinical", CLINICAL_COLUMNS)
    if "B" in args.modality.upper():
        add_modality("B", "biospecimen", BIOSPECIMEN_COLUMNS)

    if not data_dict:
        raise ValueError("adni_demo supports modality C and/or B. Try --modality CB.")

    combination_to_index = get_modality_combinations(args.modality.upper())
    data_dict["modality_comb"] = [
        combination_to_index.get("".join(sorted(set(comb))), -1)
        for comb in modality_combinations
    ]

    if args.use_common_ids and common_idx_list:
        common_idxs = set.intersection(*common_idx_list)
        train_ids = list(common_idxs & set(train_ids))
        valid_ids = list(common_idxs & set(valid_ids))
        test_ids = list(common_idxs & set(test_ids))

    mc_num_to_mc = {v: k for k, v in combination_to_index.items()}
    mc_idx_dict = {
        mc_num_to_mc[mc_num]: list(
            np.where(np.array(data_dict["modality_comb"]) == mc_num)[0]
        )
        for mc_num in set(data_dict["modality_comb"])
        if mc_num != -1
    }

    return (
        data_dict,
        encoder_dict,
        labels,
        list(train_ids),
        list(valid_ids),
        list(test_ids),
        n_labels,
        input_dims,
        transforms,
        masks,
        observed_idx_arr,
        mc_idx_dict,
        mc_num_to_mc,
    )
