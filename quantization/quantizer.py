import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST
from quantization.range_setting import max_min_range_settings, MSE_range_setting, CE_range_setting

def max_min_asym_quan(input_float, bit_width, zero_point, print_info = False) -> torch.tensor:
    lower_bound, upper_bound = max_min_range_settings(input_float)
    scale_factor = abs(upper_bound - lower_bound) / (2 ** bit_width - 1)
    input_quan = torch.clamp(torch.round(input_float / scale_factor) + zero_point, 0, 2 ** bit_width - 1)
    input_hat = scale_factor * (input_quan - zero_point)

    if print_info:
        print(f"============== Quantization Info ==============")
        print(f"Range of the input: [{input_float.min().item()}, {input_float.max().item()}]")
        print(f"Scheme for range setting: Max-min")
        print(f"Scaled factor: {scale_factor}")
        print(f"Ranges for quantization: [{lower_bound}, {upper_bound}]\n")
    return input_hat

def mse_asym_quan(input_float, bit_width, zero_point, print_info = False) -> torch.tensor:
    lower_bound, upper_bound = MSE_range_setting(input_float, bit_width)
    scale_factor = abs(upper_bound - lower_bound) / (2 ** bit_width - 1)
    input_quan = torch.clamp(torch.round(input_float / scale_factor) + zero_point, 0, 2 ** bit_width - 1)
    input_hat = scale_factor * (input_quan - zero_point)

    if print_info:
        print(f"============== Quantization Info ==============")
        print(f"Range of the input: [{input_float.min().item()}, {input_float.max().item()}]")
        print(f"Scheme for range setting: MSE")
        print(f"Scaled factor: {scale_factor}")
        print(f"Ranges for quantization: [{lower_bound}, {upper_bound}]\n")
    return input_hat

def cross_entropy_asym_quan(input_float,input_image, model, bit_width, zero_point, print_info = False) -> torch.tensor:
    if model is None:
        raise ValueError("Model must be provided for Cross Entropy range setting method")
    elif input_image is None:
        raise ValueError("Input image muse be provided for Cross Entropy range setting method")
    
    

    lower_bound, upper_bound = CE_range_setting(input_image, bit_width, model)
    scale_factor = abs(upper_bound - lower_bound) / (2 ** bit_width - 1)
    input_quan = torch.clamp(torch.round(input_float / scale_factor) + zero_point, 0, 2 ** bit_width - 1)
    input_hat = scale_factor * (input_quan - zero_point)

    if print_info:
        print(f"============== Quantization Info ==============")
        print(f"Range of the input: [{input_float.min().item()}, {input_float.max().item()}]")
        print(f"Scheme for range setting: Cross Entropy")
        print(f"Scaled factor: {scale_factor}")
        print(f"Ranges for quantization: [{lower_bound}, {upper_bound}]\n")
    return input_hat