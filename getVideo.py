import requests
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import face_recognition
from torch.utils.data import DataLoader, Dataset
from torch import nn
import os
import glob
from PIL import Image as pImage
import time

#Get video from the recording
url = "https://api.videosdk.live/v2/recordings"
headers = {'Authorization' : 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcGlrZXkiOiIzYWU1NjQ3NC05YTZiLTQxYzktYmZjOS0xZDE5ZDRjNmYwNTAiLCJwZXJtaXNzaW9ucyI6WyJhbGxvd19qb2luIl0sImlhdCI6MTcyMTcyNzU3OSwiZXhwIjoxNzIxODEzOTc5fQ.DZr17fssZ5xHd4fJG6N2AOW7br2O51-9GFw-AU_CrPI','Content-Type' : 'application/json'}
response = requests.request("GET", url,headers = headers)
print(response.text)

file_url = response.json()['data'][0]['file']['fileUrl']
print(f"File URL: {file_url}")

video_response = requests.get(file_url)
video_path = "downloaded_video.mp4"


with open(video_path, 'wb') as file:
    file.write(video_response.content)

print(f"Video downloaded and saved as {video_path}")

# Predict whether deepfake or not
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sequence_length = 60
video_file_path = r'downloaded_video.mp4'  
model_path_local = r'models'


# Define transformations
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Define the model
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)

        print(f"Extracting frames from video: {video_path}")

        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except IndexError:
                print(f"No face detected in frame {i}")
                continue
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video {video_path}")

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image
            else:
                print("Frame extraction failed")
        vidObj.release()

# Load the model
def load_model(sequence_length, num_classes=2):
    model = Model(num_classes)
    model_path = get_accurate_model(sequence_length, model_path_local)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Prediction function
def predict(model, dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for img in loader:
        img = img.squeeze(0).to(device)  # Remove the extra dimension
        img = img.unsqueeze(0)  # Ensure the shape is (1, seq_length, channels, height, width)
        print(f"Frames shape before model: {img.shape}")  # Debugging: Print frames shape
        fmap, logits = model(img)
        sm = nn.Softmax(dim=1)
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
        output = "REAL" if prediction.item() == 1 else "FAKE"
        print(f"Prediction: {output}, Confidence: {confidence:.2f}%")

def get_accurate_model(sequence_length, model_directory=r'/models'):
    model_files = glob.glob(os.path.join(model_directory, "*.pt"))
    best_model = None
    highest_accuracy = 0

    for model_file in model_files:
        model_name = os.path.basename(model_file)
        try:
            # Assuming the filename format is 'model_<accuracy>_acc_<sequence_length>_frames_*.pt'
            parts = model_name.split('_')
            accuracy = float(parts[1])  # Assuming the second part is the accuracy
            seq_len = int(parts[3])     # Assuming the fourth part is the sequence length

            if seq_len == sequence_length and accuracy > highest_accuracy:
                best_model = model_file
                highest_accuracy = accuracy
        except (IndexError, ValueError) as e:
            print(f"Skipping file {model_name} due to parsing error: {e}")

    if best_model:
        return best_model
    else:
        raise ValueError("No model found for the specified sequence length.")

def main():
    start_time = time.time()

    dataset = validation_dataset([video_file_path], sequence_length, transform=train_transforms)

    model = load_model(sequence_length)

    predict(model, dataset)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()