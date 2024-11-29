import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST

def model_test(model) -> float:
    model.eval()
    device = torch.device("cpu")
    loss_fn = nn.CrossEntropyLoss()
    test_kwargs = {'batch_size': 1000}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

    test_dataset = FashionMNIST('model/data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    # print(
    #     f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
    #     f" ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    # )
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy