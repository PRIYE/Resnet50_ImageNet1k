import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.classifier import DogBreedClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()

    # Load model
    model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path)
    model.eval()

    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=args.input_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Evaluate
    model.val_acc.reset()
    for batch in dataloader:
        images, labels = batch
        with torch.no_grad():
            outputs = model(images)
            model.val_acc(outputs, labels)

    print(f"Validation Accuracy: {model.val_acc.compute():.4f}")

if __name__ == "__main__":
    main()
