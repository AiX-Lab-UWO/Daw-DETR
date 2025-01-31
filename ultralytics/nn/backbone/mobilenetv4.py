from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import clip
from torchvision import transforms

__all__ = ['MobileNetV4ConvSmall']

MNV4ConvSmall_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 32, 3, 2],
            [32, 32, 1, 1]
        ]
    },
    "layer2": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 96, 3, 2],
            [96, 64, 1, 1]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            [64, 96, 5, 5, True, 2, 3],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 3, 0, True, 1, 4],
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            [96, 128, 3, 3, True, 2, 6],
            [128, 128, 5, 5, True, 1, 4],
            [128, 128, 0, 5, True, 1, 4],
            [128, 128, 0, 5, True, 1, 3],
            [128, 128, 0, 3, True, 1, 4],
            [128, 128, 0, 3, True, 1, 4],
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [128, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}

MODEL_SPECS = {
    "MobileNetV4ConvSmall": MNV4ConvSmall_BLOCK_SPECS,
}


def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1))
        self.block.add_module('conv_3x3',
                              conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 middle_dw_downsample,
                 stride,
                 expand_ratio
                 ):
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
        x = self._expand_conv(x)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
        x = self._proj_conv(x)
        return x


def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"convbn_{i}", conv_2d(**args))
    elif block_names == "uib":
        schema_ = ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride',
                   'expand_ratio']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers


class ClipFeatureFusionModule(nn.Module):
    def __init__(self, mobilenet_feat_dim, output_dim):
        super(ClipFeatureFusionModule, self).__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.device = next(self.clip_model.parameters()).device
        self.text_transform = nn.Linear(512, 512).to(self.device)

        self.fusion_transform = nn.Sequential(
            nn.Conv2d(2304, output_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(self.device)

        self.adaptive_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False).to(self.device)

        self.to_pil = transforms.ToPILImage()
        self.norm_mobilenet = nn.BatchNorm2d(mobilenet_feat_dim, eps=1e-5, momentum=0.1).to(self.device)
        self.norm_clip = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1).to(self.device)
        self.norm_text = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1).to(self.device)
        self.mobilenet_feat_dim = mobilenet_feat_dim

    def forward(self, original_image, mobilenet_features):
        batch_size, channels, height, width = mobilenet_features.shape
        # original_image_pil = self.to_pil(original_image[0].cpu())
        # original_image_processed = self.preprocess(original_image_pil).unsqueeze(0).to(self.device)
        original_image_processed = torch.stack([
            self.preprocess(self.to_pil(original_image[i].cpu())) for i in range(batch_size)
        ]).to(self.device)

        if self.norm_mobilenet.num_features != mobilenet_features.shape[1]:
            self.norm_mobilenet = nn.BatchNorm2d(mobilenet_features.shape[1], eps=1e-5, momentum=0.1).to(self.device)

        mobilenet_features = self.norm_mobilenet(mobilenet_features)

        input_text = [
            "weather: rainy", "weather: snowy", "weather: clear", "weather: overcast",
            "weather: partly cloudy", "weather: foggy", "weather: sandstorm",
            "scene: tunnel", "scene: residential", "scene: parking lot",
            "scene: city street", "scene: gas stations", "scene: highway",
            "timeofday: daytime", "timeofday: night", "timeofday: dawn/dusk"
        ]
        categories = ["weather:", "scene:", "timeofday:"]

        text_tokens = clip.tokenize(input_text).to(self.device)
        with torch.no_grad():
            clip_image_features = self.clip_model.encode_image(original_image_processed)
            clip_text_features = self.clip_model.encode_text(text_tokens).float()

        best_text_features = self.get_best_matching_text_features(clip_image_features, clip_text_features, input_text,
                                                                  categories)

        transformed_text_features = self.text_transform(best_text_features).to(self.device)

        # print(f"Before transformation: transformed_text_features.shape = {transformed_text_features.shape}")

        transformed_text_features = transformed_text_features.view(batch_size, -1, 1, 1)

        transformed_text_features = transformed_text_features.expand(batch_size, -1, mobilenet_features.shape[2],
                                                                     mobilenet_features.shape[3])

        clip_image_features = clip_image_features.view(batch_size, -1, 1, 1)
        clip_image_features = clip_image_features.expand(batch_size, -1, mobilenet_features.shape[2],
                                                         mobilenet_features.shape[3])

        # print(f"After transformation: transformed_text_features.shape = {transformed_text_features.shape}")
        # print(f"After transformation: clip_image_features.shape = {clip_image_features.shape}")
        # print(f"mobilenet_features.shape = {mobilenet_features.shape}")

        clip_image_features = self.norm_clip(clip_image_features)
        transformed_text_features = self.norm_text(transformed_text_features)

        combined_features = torch.cat([mobilenet_features, clip_image_features, transformed_text_features], dim=1)

        combined_channels = combined_features.shape[1]
        expected_channels = mobilenet_features.shape[1]

        if self.adaptive_conv.in_channels != combined_channels:
            self.adaptive_conv = nn.Conv2d(combined_channels, expected_channels, kernel_size=1, bias=False).to(
                self.device)

        fused_features = self.fusion_transform(combined_features)
        # print(f"fused_features Channels: {fused_features.shape[1]}")

        return fused_features

    def get_best_matching_text_features(self, image_features, text_features, input_text, categories):
        image_features = image_features.float()
        batch_size = image_features.shape[0]

        similarities = image_features @ text_features.T
        best_text_features = []

        for category in categories:
            category_indices = [i for i, text in enumerate(input_text) if text.startswith(category)]

            if not category_indices:  # Handle empty case
                # print(f"Warning: No matching text for category '{category}'")
                continue

            category_similarities = similarities[:, category_indices]

            best_indices = category_similarities.argmax(dim=1)

            best_texts = text_features[torch.tensor(category_indices, device=self.device)[best_indices]]
            best_text_features.append(best_texts)

        if not best_text_features:
            # print("Warning: No valid text features were matched.")
            return torch.zeros((batch_size, text_features.shape[1]), device=self.device)

        best_text_features = torch.stack(best_text_features, dim=1).mean(dim=1)

        return best_text_features.to(self.device)


class MobileNetV4(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert model in MODEL_SPECS.keys()
        self.model = model
        self.spec = MODEL_SPECS[self.model]

        self.conv0 = build_blocks(self.spec['conv0'])
        self.layer1 = build_blocks(self.spec['layer1'])
        self.layer2 = build_blocks(self.spec['layer2'])
        self.layer3 = build_blocks(self.spec['layer3'])
        self.layer4 = build_blocks(self.spec['layer4'])
        self.layer5 = build_blocks(self.spec['layer5'])
        self.features = nn.ModuleList([self.conv0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5])

        self.clip_fusion_module = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            output_features = self.forward(dummy_input)
            self.channel = [f.size(1) if f is not None else None for f in output_features]

    def forward(self, x):
        original_image = x.clone()
        input_size = x.size(2)
        scale = [4, 8, 16, 32]
        features = [None, None, None, None]

        for f in self.features:
            x = f(x)
            if input_size // x.size(2) in scale:
                features[scale.index(input_size // x.size(2))] = x

        mobilenet_features = x.to(self.device)
        mobilenet_feat_dim = mobilenet_features.shape[1]

        if self.clip_fusion_module is None:
            self.clip_fusion_module = ClipFeatureFusionModule(mobilenet_feat_dim=mobilenet_feat_dim,
                                                              output_dim=mobilenet_feat_dim).to(self.device)
        elif self.clip_fusion_module.mobilenet_feat_dim != mobilenet_feat_dim:
            self.clip_fusion_module = ClipFeatureFusionModule(mobilenet_feat_dim=mobilenet_feat_dim,
                                                              output_dim=mobilenet_feat_dim).to(self.device)

        fused_features = self.clip_fusion_module(original_image=original_image, mobilenet_features=mobilenet_features)
        features[-1] = fused_features
        # print("Final Feature Shapes:")
        # for i, f in enumerate(features):
        #     if f is not None:
        #         print(f"features[{i}].shape = {f.shape}")

        return features


def MobileNetV4ConvSmall():
    model = MobileNetV4('MobileNetV4ConvSmall')
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV4ConvSmall().to(device)
    inputs = torch.randn((8, 3, 640, 640)).to(device)
    res = model(inputs)

    for i in res:
        print(i.size())
