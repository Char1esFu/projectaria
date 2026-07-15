# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchvision import transforms


def preprocess_image(et_image, size=(240, 320)):
    if et_image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale HxW eye-tracking image, got {tuple(et_image.shape)}"
        )

    _, w = et_image.shape
    if w % 2:
        raise ValueError(
            f"Expected side-by-side eye images with an even width, got width={w}"
        )

    output_h, output_w = (int(size[0]), int(size[1]))
    pred_image = torch.zeros((1, 2, output_h, output_w), dtype=torch.float32)

    pred_image[0, 0, :, :] = resize_and_normalize(et_image[:, : w // 2], size, False)
    pred_image[0, 1, :, :] = resize_and_normalize(et_image[:, w // 2 :], size, True)

    return pred_image


def resize_and_normalize(image, size=(240, 320), should_flip=False):
    image = image.float()
    value_min = torch.min(image)
    value_range = torch.max(image) - value_min
    normalized_image = (image - value_min) / value_range.clamp_min(1e-6) - 0.5
    # Flip the image
    if should_flip:
        normalized_image = torch.fliplr(normalized_image)

    target_size = (int(size[0]), int(size[1]))
    if tuple(normalized_image.shape) == target_size:
        return normalized_image

    # TorchVision expects tensor images as CxHxW. Eye images arrive as HxW,
    # so add a temporary grayscale channel for profiles such as profile5.
    return transforms.Resize(target_size, antialias=True)(
        normalized_image.unsqueeze(0)
    ).squeeze(0)
