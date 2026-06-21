import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms

from model import CLASS_NAMES, load_model


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "models/nature_cnn.pth"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cpu")
model = load_model(MODEL_PATH, device)

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

image_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_index.item()]
    confidence_score = confidence.item() * 100

    top_probs, top_indices = torch.topk(probabilities, 3)

    top_predictions = []
    for prob, index in zip(top_probs[0], top_indices[0]):
        top_predictions.append({
            "class_name": CLASS_NAMES[index.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return predicted_class, round(confidence_score, 2), top_predictions


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    top_predictions = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            filename = file.filename
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            prediction, confidence, top_predictions = predict_image(image_path)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        top_predictions=top_predictions
    )


if __name__ == "__main__":
    app.run(debug=True)