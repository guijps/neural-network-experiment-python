import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
from PIL import Image, ImageOps
from torchvision import datasets, transforms
from nn_models import ConvolutionalNN,SimpleNN

def _preprocess_image(img: Image.Image) -> Image.Image:
    #Converting to Grayscale
    img = img.convert('L')
    #??
    img = ImageOps.autocontrast(img)
    #??
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)


    w,h = img.size
    size = max(w,h)
    canvas = Image.new('L', (size, size),color=0)
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas.resize((28, 28), Image.BILINEAR)



def predict_image(image_path):
    img = Image.open(image_path)
    img = _preprocess_image(img)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    print(f'Predicted digit: {predicted.item()}')




parser = argparse.ArgumentParser(description='Neural Network Experiment')
parser.add_argument('--model', type=str, choices=['simple', 'cnn'], default='cnn', help='Choose the model to use: simple or cnn')
parser.add_argument('--retrain', action='store_true', help='Retrain the model even if a checkpoint exists')
args = parser.parse_args()
print(f"Using model: {args.model} and retrain={args.retrain}")
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1000, shuffle=False)

if args.model == 'simple':
    model = SimpleNN()
else:
    model = ConvolutionalNN()


if os.path.exists(model.getCheckPointPath()) and not args.retrain:
    print("Loading model from disk...")
    model.load_state_dict(torch.load(model.getCheckPointPath()))
    model.eval()
    print("Model loaded from disk.")
else:
    print("Training model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), model.getCheckPointPath())
    model.eval()
    print("Model trained and saved to disk.")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += ( predicted == labels).sum().item()
        total += labels.size(0)
    print(f'Test Accuracy: {100 * correct / total}%')

for i in range(1,10):
  print(f'Testing image for digit {i}:')
  predict_image(f'../test_images/{i}.png')  