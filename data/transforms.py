from torchvision import transforms

InferenceTransform = transforms.Compose(
    [
        transforms.Resize((28, 28)),  # Ensure all images are resized to 28x28
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
TrainTransform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomInvert(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
