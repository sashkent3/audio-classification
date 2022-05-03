import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as T
from scipy.special import softmax

sr = 16000
seconds = 5

print(f'Recording for {seconds} seconds...')
sample = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
sd.wait()
print('Done!')
transform = T.MelSpectrogram(n_mels=64)
model = torch.jit.load('model_scripted.pt')
model.cpu()
model.eval()
with torch.no_grad():
    pred = softmax(
        model(transform(torch.tensor(sample.T[np.newaxis])))).flatten()

print(f'Male: {pred[0] * 100:.1f}%, Female: {pred[1] * 100:.1f}%')
