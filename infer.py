import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import concurrent.futures

class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path

def gather_image_paths(root_dir):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def process_batch(batch, model, device):
    data = batch.to(device)
    output = model(data)
    preds = output.argmax(dim=1, keepdim=True)
    return preds.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Digit Classification Inference")
    parser.add_argument("--model", type=str, required=True, help="path to the trained TorchScript model")
    parser.add_argument("--target", type=str, required=True, help="directory with images for inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(args.model).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image_paths = gather_image_paths(args.target)
    dataset = InferenceDataset(image_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    digit_counts = [0] * 10

    with torch.no_grad():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, batch, model, device) for batch, _ in data_loader]
            for future in concurrent.futures.as_completed(futures):
                preds = future.result()
                for pred in preds:
                    digit_counts[pred.item()] += 1

    print("digit,count")
    for digit, count in enumerate(digit_counts):
        print(f"{digit},{count}")

if __name__ == "__main__":
    main()
