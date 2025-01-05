import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.classifier import DogBreedClassifier

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()

    # Create output directory
    Path(args.output_folder).mkdir(exist_ok=True)

    # Load model
    model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path)
    model.eval()

    # Process each image
    transform = get_transform()
    class_labels = ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German Shepherd',
                   'Golden Retriever', 'Labrador Retriever', 'Poodle', 'Rottweiler', 
                   'Yorkshire Terrier']

    for img_path in Path(args.input_folder).glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        # Save results
        result = f"{img_path.name}: {class_labels[pred_idx]} ({confidence:.2f})\n"
        with open(Path(args.output_folder) / "predictions.txt", "a") as f:
            f.write(result)

if __name__ == "__main__":
    main()
