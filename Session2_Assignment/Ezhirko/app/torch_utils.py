import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image
from app.resnet import ResNet18

def transform_image(image_bytes):
    transform = transforms.Compose([
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    image = Image.open(io.BytesIO(image_bytes))
    image.save("app/static/uploads/upload.png");
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    # images = image_tensor.reshape(-1, 32*32)
    outputs = model(image_tensor)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted


model = ResNet18()

PATH = "app/cifar_resnet18.pth"
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()