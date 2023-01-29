"""This module implements a general Experiment class."""

import os
import time
from dataclasses import dataclass
from typing import Any, cast

import torch
import yaml
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

from experitorch.auto_loading import get_model_class, get_model_parameters, get_model_trainer
from experitorch.color_logging import LoggerMixin, get_logger


@dataclass(frozen=True)
class PathGenerator:
    """Helper class to generate the paths of an experiment."""

    root: str

    def _concat_root_and_suffix(self, suffix) -> str:
        return os.path.join(self.root, suffix)

    @property
    def config_path(self) -> str:
        return self._concat_root_and_suffix("config.yaml")

    @property
    def checkpoint_path(self) -> str:
        return self._concat_root_and_suffix("checkpoints")

    @property
    def results_path(self) -> str:
        return self._concat_root_and_suffix("results")

    @property
    def tensorboard_path(self) -> str:
        return self._concat_root_and_suffix("tensorboard")

    @property
    def figures_path(self) -> str:
        return self._concat_root_and_suffix("figures")

    def get_path_by_name(self, name: str) -> str:
        return getattr(self, "_".join([name, "path"]))


@dataclass(frozen=True)
class PathCreator:
    """Helper class to create all the paths of an experiment."""

    def __init__(self, root: str):
        self.path_generator = PathGenerator(root)

    def create_config_file(self, config: dict):
        with open(self.path_generator.config_path, "w") as file:
            yaml.dump(config, file)

    def create_checkpoits_dir(self):
        os.makedirs(self.path_generator.checkpoints_path, exist_ok=True)

    def create_results_dir(self):
        os.makedirs(self.path_generator.results_path, exist_ok=True)

    def create_tensorboard_dir(self):
        os.makedirs(self.path_generator.tensorboard_path, exist_ok=True)

    def create_figures_dir(self):
        os.makedirs(self.path_generator.figures_path, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig(BaseModel):
    """This class implements the configuration of an experiment."""

    project: str
    model_type: str
    parameters: Any


@dataclass(frozen=True)
class Experiment(LoggerMixin):
    """This class implements a general experiment."""

    def __init__(self, path: str, config: dict):
        self.path = path
        self.config = config

        self.paths = PathGenerator(path)

        project, model_type = config["project"], config["model_type"]

        self.model_class = get_model_class(project, model_type)
        self.trainer_class = get_model_trainer(project, model_type)
        self.parameters_class = get_model_parameters(project, model_type)

    @staticmethod
    def open(path: str) -> "Experiment":
        """Load the experiment from a directory."""
        logger = get_logger("Experiment.from_directory")
        logger.info(f"Loading experiment from {path}")
        with open(path, "r") as f:
            config = cast(dict, ExperimentConfig(**yaml.load(f, Loader=yaml.SafeLoader)).dict(by_alias=True))
        return Experiment(path, config)

    @staticmethod
    def create_from_dict(config: dict, outdir: str = ".") -> "Experiment":
        logger = get_logger("Experiment.create_from_dict")
        logger.info(f"Creating an experiment with config {config}")
        name = Experiment.generate_new_name(config)
        path = os.path.join(outdir, name)
        if os.path.exists(path):
            logger.error("Path %s already exists", path)
            raise FileExistsError(f"Path {path} already exists")
        else:
            os.mkdir(path)
            path_creator = PathCreator(path)
            path_creator.create_config_file(config)
            path_creator.create_checkpoits_dir()
            path_creator.create_results_dir()
            path_creator.create_tensorboard_dir()
            path_creator.create_figures_dir()
        return Experiment(path, config)

    @staticmethod
    def create_from_yaml(path: str, outdir: str = ".") -> "Experiment":
        logger = get_logger("Experiment.create_from_yaml")
        logger.info(f"Creating an experiment from {path} file")
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return Experiment.create_from_dict(config, outdir)

    @staticmethod
    def generate_new_name(config: dict) -> str:
        """Generate the name of the experiment."""
        project, model_type = config["project"], config["model_type"]
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        name = f"{project}-{model_type}-{timestamp}"
        return name

    def is_checkpoint_available(self, name: str) -> bool:
        """checks if a checkpoint is in the experiment directory."""
        return self._is_element_available("checkpoints", name)

    def is_results_available(self, name: str) -> bool:
        """checks if a results is in the experiment directory."""
        return self._is_element_available("results", name)

    def load_checkpoint(self, name: str) -> dict:
        """Loads a checkpoint."""
        checkpoint = cast(dict, self._load_element("checkpoints", name))
        return checkpoint

    def save_checkpoint(self, checkpoint: dict, name: str):
        """Saves a checkpoint."""
        self._save_element(checkpoint, "checkpoints", name)

    def load_results(self, name: str) -> dict:
        """Loads a results."""
        results = cast(dict, self._load_element("results", name))
        return results

    def save_results(self, results: dict, name: str):
        """Saves a results."""
        self._save_element(results, "results", name)

    def get_tensorboard_writer(self, name: str) -> SummaryWriter:
        """Creates a tensorboardX writer."""
        path = self.paths.tensorboard_path
        if name is not None:
            path = os.path.join(path, name)
        writer = SummaryWriter(path)
        return writer

    def _create_element_path(self, element_name: str, name: str) -> str:
        """Creates an element path from its name and suffix."""
        try:
            path = self.paths.get_path_by_name(element_name)
        except AttributeError:
            self.logger.error("Element %s not found", element_name)
            raise
        path = os.path.join(path, name)
        return path

    def _is_element_available(self, element_name: str, name: str) -> bool:
        """Checks if an element is available to load"""
        path = self._create_element_path(element_name, name)
        return os.path.isfile(path)

    def _load_element(self, element_name: str, name: str) -> Any:
        """Loads an element from its name and suffix."""
        path = self._create_element_path(element_name, name)
        element = torch.load(path, map_location=lambda storage, loc: storage)
        return element

    def _save_element(self, element: Any, element_name: str, name: str):
        """Saves an element from its name and suffix"""
        path = self._create_element_path(element_name, name)
        torch.save(element, path)
