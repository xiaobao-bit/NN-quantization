import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
from quantization.quantizer import max_min_asym_quan, mse_asym_quan, cross_entropy_asym_quan
from utils.utils import model_test


def model_quantizer(model, bit_width, zero_point, quantizer, model_path, quantization_choice='W', save_path=None):
    """
    Quantizes a model's weights or activations.

    Args:
        model: PyTorch model to quantize.
        bit_width: Bit-width for quantization.
        zero_point: Zero point for quantization.
        quantizer: Quantization method ('CE', 'MSE', 'MM').
        model_path: Path to the pre-trained model.
        quantization_choice: 'W' for weight quantization, 'A' for activation quantization.
        save_path: Path to save the quantized model. If None, the model won't be saved.
    """
    if quantization_choice not in ['W', 'A']:
        raise ValueError(f"Unsupported quantization choice: {quantization_choice}")
    if quantizer not in ['CE', 'MM', 'MSE']:
        raise ValueError(f"Unsupported quantizer: {quantizer}")

    # Load the pre-trained model
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Prepare input image for CE quantization
    if quantizer == 'CE':
        test_kwargs = {'batch_size': 1}  # Only need one sample
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_dataset = FashionMNIST('model/data', train=False, download=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        input_image, _ = next(iter(test_loader))

    # Quantize weights
    if quantization_choice == 'W':
        for _, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.weight is not None:
                    if quantizer == 'CE':
                        module.weight.data = cross_entropy_asym_quan(
                            module.weight, input_image, model, bit_width, zero_point
                        )
                    elif quantizer == 'MM':
                        module.weight.data = max_min_asym_quan(module.weight, bit_width, zero_point)
                    elif quantizer == 'MSE':
                        module.weight.data = mse_asym_quan(module.weight, bit_width, zero_point)

                if module.bias is not None:
                    if quantizer == 'CE':
                        module.bias.data = cross_entropy_asym_quan(
                            module.bias, input_image, model, bit_width, zero_point
                        )
                    elif quantizer == 'MM':
                        module.bias.data = max_min_asym_quan(module.bias, bit_width, zero_point)
                    elif quantizer == 'MSE':
                        module.bias.data = mse_asym_quan(module.bias, bit_width, zero_point)

    # Quantize activations
    elif quantization_choice == 'A':
        if quantizer == 'CE':
            register_activation_CE_hooks(model, input_image)
        else:
            for _, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU)):
                    if quantizer == 'MM':
                        module.register_forward_hook(activation_MM_quan_hook)
                    elif quantizer == 'MSE':
                        module.register_forward_hook(activation_MSE_quan_hook)

    # Save the quantized model if a save path is provided
    if save_path:
        torch.save(model.state_dict(), save_path)

    # Evaluate the quantized model
    test_loss, accuracy = model_test(model)
    return test_loss, accuracy


def activation_MM_quan_hook(module, input, output):
    bit_width = 8
    zero_point = 1
    return max_min_asym_quan(output, bit_width, zero_point)


def activation_MSE_quan_hook(module, input, output):
    bit_width = 8
    zero_point = 1
    return mse_asym_quan(output, bit_width, zero_point)


def activation_CE_quan_hook(module, input, output, model, input_image):
    bit_width = 8
    zero_point = 1
    return cross_entropy_asym_quan(output, input_image, model, bit_width, zero_point)


def create_activation_CE_hook(model, input_image):
    def activation_CE_hook(module, input, output):
        bit_width = 8
        zero_point = 1
        return cross_entropy_asym_quan(output, input_image, model, bit_width, zero_point)
    return activation_CE_hook


def register_activation_CE_hooks(model, input_image):
    ce_hook = create_activation_CE_hook(model, input_image)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
            module.register_forward_hook(ce_hook)