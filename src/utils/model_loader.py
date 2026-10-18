from huggingface_hub import hf_hub_download
import os

from networkx import is_digraphical
from src.models.vit import VisionTransformer
from src.models.predictor import (
    ViTPredictor
)
from typing import Any
import torch
import torch.nn as nn
import shutil
from pathlib import Path


ROOT_OUT_FOLDER: str = ".models"

def load_pretrained_backbone_models(
    geometry: str, 
    run_id: str, 
    student_model: VisionTransformer, 
    teacher_model: VisionTransformer, 
    predictor: ViTPredictor, 
    logger,
    device,
    epoch_cp = 20
    ):

    MODEL_OUT_FOLDER: str = f"{ROOT_OUT_FOLDER}/{geometry}"

    BACKBONE_MODEL_MAP: dict[str, nn.Module] = {
        f"teacher_model_jepa_{run_id}_{str(epoch_cp)}ep.pth": teacher_model,
        f"trained_student_jepa_{run_id}_{str(epoch_cp)}ep.pth": student_model,
        f"trained_predictor_jepa_{run_id}_{str(epoch_cp)}ep.pth": predictor
    }

    logger.info(f"Downloading {geometry} pretrained JEPA models into {MODEL_OUT_FOLDER}...")

    for model_name, model_type in BACKBONE_MODEL_MAP.items():
        logger.info(f"Downloading {model_name}...")
        path = hf_hub_download(
            repo_id="csanadl/local-ijepa-geometry",
            filename=model_name,
            local_dir=MODEL_OUT_FOLDER,
        )

        logger.info(f"Model downloaded successfully: {path}")
        logger.info("Loading model into memory...")

        model_type.load_state_dict(torch.load(path, map_location=device))

        logger.info(f"{model_name} successfully loaded into memory!")


def load_pretrained_finetune_models(
    geometry: str, 
    run_id: str, 
    student_model: VisionTransformer, 
    teacher_model: VisionTransformer, 
    predictor: ViTPredictor,
    block_predictor,
    logger,
    device,
    epoch_cp = 20
    ):

    MODEL_OUT_FOLDER: str = f"{ROOT_OUT_FOLDER}/{geometry}"

    FINETUNE_MODEL_MAP: dict[str, nn.Module] = {
        f"teacher_model_{geometry}_{run_id}_{str(epoch_cp)}ep.pth": teacher_model,
        f"trained_student_{geometry}_{run_id}_{str(epoch_cp)}ep.pth": student_model,
        f"trained_predictor_{geometry}_{run_id}_{str(epoch_cp)}ep.pth": predictor,
        f"trained_block_pred_{geometry}_{run_id}_{str(epoch_cp)}ep.pth": block_predictor
    }

    logger.info(f"Downloading {geometry} pretrained finetuning models into {MODEL_OUT_FOLDER}...")

    for model_name, model_type in FINETUNE_MODEL_MAP.items():
        logger.info(f"Downloading {model_name}...")
        path = hf_hub_download(
            repo_id="csanadl/local-ijepa-geometry",
            filename=model_name,
            local_dir=MODEL_OUT_FOLDER,
        )

        logger.info(f"Model downloaded successfully: {path}")
        logger.info("Loading model into memory...")

        model_type.load_state_dict(torch.load(path, map_location=device))

        logger.info(f"{model_name} successfully loaded into memory!")

def clear_ws(logger):
    geometries: list[str] = ["euclidean", "hyperbolic"]

    logger.info("clearing workspace...")

    for geometry in geometries:
        folder = Path(ROOT_OUT_FOLDER) / geometry
        if not folder.is_dir():
            continue
        for child in folder.iterdir():
            if child.name == "README.md": # dont delete, only a placeholder
                continue
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    logger.info("workspace cleared successfully!")