"""This module contains functions to load the experiment parts."""

import importlib
import inspect
from typing import cast

import torch.nn as nn
from pydantic import BaseModel

from experitorch.color_logging import get_logger


def get_model_component_class(project: str, model_type: str, component_type: str, component_name: str) -> object:
    """Finds the model component class."""
    logger = get_logger("config.get_model_component_class")

    model_type = model_type.lower()
    component_type = component_type.lower()
    component_name = component_name.lower()

    model_module_path = f"{project}.models.{model_type}"
    component_module_path = f"{project}.models.{model_type}.{component_type}"

    # Checks it can import the the ml model.
    try:
        _ = importlib.import_module(model_module_path)
    except ModuleNotFoundError:
        logger.error(f'Cannot import "{model_type}". Make sure the model exists')
        raise ModuleNotFoundError

    # Checks if can import the model component and if it is, loads it.
    try:
        component_module = importlib.import_module(component_module_path)
        component_module_members = inspect.getmembers(component_module)
    except ModuleNotFoundError:
        logger.error(f'The "{model_type}" has no component {component_type}')
        raise ModuleNotFoundError

    component_members_header = [name.lower() for name, _ in component_module_members]

    # Checks if the component exists in the component module.
    if component_name not in component_members_header:
        logger.error(f'The "{component_name}" model component does not exist.')
        raise AttributeError

    member_idx = component_members_header.index(component_name)
    component_name_right_case = component_module_members[member_idx][0]
    component_class = getattr(component_module, component_name_right_case)

    # TODO: check if necessary, but seems useless and preventive...
    # checks if the class is really in model.py
    if inspect.getmodule(component_class) != component_module:
        warning_str = f'The "{component_name}" is extern to the model.py module'
        logger.warning(warning_str)

    return component_class


def get_model_class(project: str, model_type: str) -> object:
    """Finds the model class from a ml model type."""
    logger = get_logger("config.get_model_class")

    model_class = get_model_component_class(project, model_type, "model", model_type)

    def search_for_pytorch(current_class):
        for base in current_class.__bases__:
            if base == nn.Module:
                return True
            if search_for_pytorch(base):
                return True
        return False

    # Checks if the class is really a nn.Module class.
    if not search_for_pytorch(model_class):
        logger.warning(f'The "{model_type}" model is not a PyTorch module')

    return model_class


def get_model_trainer(project: str, model_type: str) -> object:
    """Finds the model tainer class from a ml model type."""
    trainer_name = model_type + "trainer"
    trainer_class = get_model_component_class(project, model_type, "trainer", trainer_name)
    return trainer_class


def get_model_parameters(project: str, model_type: str) -> BaseModel:
    """Finds the model parameters class from a ml model type."""
    parameters_name = model_type + "parameters"
    parameters_class = cast(BaseModel, get_model_component_class(project, model_type, "config", parameters_name))
    return parameters_class
