
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image

from Model.Cancer_Detector import CancerCNN

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = CancerCNN(num_classes=2, metadata_features=0)
state_dict = torch.load("Cancter_Detector.pt", map_location=device)

# handle both full checkpoint or pure state_dict
if "model_state_dict" in state_dict:
    model.load_state_dict(state_dict["model_state_dict"])
else:
    model.load_state_dict(state_dict)

model.to(device)
model.eval()

API_KEY = "password"

app = FastAPI()

class PredictionRequest(BaseModel):
    image_path: str
    api_key: str

preprocess = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

@app.post("/predict")
def predict(req: PredictionRequest):

    if req.api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        img = Image.open(req.image_path).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image load failed: {e}")

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    return {
        "prediction": int(predicted.item()),
        "label_meaning": "1 = Tumor, 0 = Normal"
    }
