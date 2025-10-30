from typing import Dict
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import TensorDict

__all__ = ["CustomExtractor", "CustomExtractorResNet18", "CustomExtractorResNet34"]

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        mlp_layers: list = [4, 4],
    ) -> None:
        super().__init__(observation_space, features_dim=1)
        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(
                    subspace,
                    features_dim=cnn_output_dim,
                    normalized_image=normalized_image,
                )
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Sequential()
                init_dim = get_flattened_obs_dim(subspace)
                for layer in mlp_layers:
                    extractors[key].add_module(
                        f"layer_{len(extractors[key])}", nn.Linear(init_dim, layer)
                    )
                    extractors[key].add_module(name="relu", module=nn.ReLU())
                    init_dim = layer

                total_concat_size += init_dim

        self.extractors = nn.ModuleDict(extractors)

        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


class CustomExtractorResNet18(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        mlp_layers: list = [16, 4],
    ) -> None:
        super().__init__(observation_space, features_dim=1)
        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                in_ch = subspace.shape[0]
                model = resnet18(ResNet18_Weights.IMAGENET1K_V1)
                num_ftrs = model.fc.in_features
                model.fc = nn.Sequential(*[
                    nn.Linear(num_ftrs, cnn_output_dim),
                    nn.Linear(cnn_output_dim, cnn_output_dim),
                ])
                model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                extractors[key] = model
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Sequential()
                init_dim = get_flattened_obs_dim(subspace)
                for layer in mlp_layers:
                    extractors[key].add_module(
                        f"layer_{len(extractors[key])}", nn.Linear(init_dim, layer)
                    )
                    extractors[key].add_module(name="relu", module=nn.ReLU())
                    init_dim = layer

                total_concat_size += init_dim
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


class CustomExtractorResNet34(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        mlp_layers: list = [16, 4],
    ) -> None:
        super().__init__(observation_space, features_dim=1)
        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                in_ch = subspace.shape[0]
                model = resnet34(ResNet34_Weights.IMAGENET1K_V1)
                num_ftrs = model.fc.in_features
                model.fc = nn.Sequential(*[
                    nn.Linear(num_ftrs, cnn_output_dim),
                    nn.Linear(cnn_output_dim, cnn_output_dim),
                ])
                model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                extractors[key] = model
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Sequential()
                init_dim = get_flattened_obs_dim(subspace)
                for layer in mlp_layers:
                    extractors[key].add_module(
                        f"layer_{len(extractors[key])}", nn.Linear(init_dim, layer)
                    )
                    extractors[key].add_module(name="relu", module=nn.ReLU())
                    init_dim = layer

                total_concat_size += init_dim
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


















