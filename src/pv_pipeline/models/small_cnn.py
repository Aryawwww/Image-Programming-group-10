from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        architecture = config["architecture"]
        blocks = []
        for block in architecture["blocks"]:
            blocks.extend(
                (
                    nn.Conv2d(
                        int(block["in_channels"]),
                        int(block["out_channels"]),
                        kernel_size=tuple(architecture["convolution_kernel"]),
                        stride=tuple(architecture["convolution_stride"]),
                        padding=tuple(architecture["convolution_padding"]),
                        bias=bool(architecture["convolution_bias"]),
                    ),
                    nn.BatchNorm2d(
                        int(block["out_channels"]),
                        eps=float(architecture["batch_norm_epsilon"]),
                        momentum=float(architecture["batch_norm_momentum"]),
                        affine=bool(architecture["batch_norm_affine"]),
                        track_running_stats=bool(
                            architecture["batch_norm_track_running_stats"]
                        ),
                    ),
                    nn.ReLU(inplace=bool(architecture["relu_inplace"])),
                    nn.MaxPool2d(
                        kernel_size=tuple(architecture["max_pool_kernel"]),
                        stride=tuple(architecture["max_pool_stride"]),
                    ),
                )
            )
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(float(architecture["dropout_probability"]))
        final_channels = int(architecture["blocks"][-1]["out_channels"])
        self.classifier = nn.Linear(final_channels, int(architecture["output_classes"]))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(1)
        return self.classifier(self.dropout(pooled))
