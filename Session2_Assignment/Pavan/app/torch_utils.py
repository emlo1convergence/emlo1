import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image
from app.resnet import ResNet18


def transform_image(image):
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((32,32)),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    # images = image_tensor.reshape(-1, 32*32)
    outputs = model(image_tensor)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted


model = ResNet18()

PATH = "app/cifar_resnet18.pth"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# model.to(device)
# model.load_state_dict(torch.load(PATH))
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
