import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

print("Loading models...")

mtcnn = MTCNN(image_size=160)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

dummy = torch.randn(1, 3, 160, 160)

with torch.no_grad():
    emb = resnet(dummy)

print("Embedding shape:", emb.shape)
print("SUCCESS ✅")
