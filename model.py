import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torchvision import models

class Last_Decoder(nn.Module):
    def __init__(self, inputsize = 2048, n_layers=2, h_size=512):
        super(Last_Decoder, self).__init__()
        self.lstm = nn.LSTM(inputsize, h_size, dropout=0.2, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(h_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, c):
        lstm, _ = self.lstm(x, c)
        logit = self.fc(lstm[-1])
        return logit

class All_Decoder(nn.Module):
    def __init__(self, inputsize = 2048, n_layers=2, h_size=512):
        super(All_Decoder, self).__init__()
        self.lstm = nn.LSTM(16384, h_size, dropout=0.2, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(h_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, c):
        lstm, _ = self.lstm(x, c)
        logit = self.fc(lstm)
        return logit

class ConvLSTM(nn.Module):
    def __init__(self, n_layers=2, h_size=512, encoder_name="Resnet50"):
        super(ConvLSTM, self).__init__()
        self.h_size = h_size
        self.n_layers = n_layers
        self.encoder = self._encoder(encoder_name)
        self.decoder = Last_Decoder(self.decoder_size, n_layers, h_size)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr = 0.001)

    def _encoder(self, model_name):
        available_models = {
            "Resnet50": {
                'model': models.resnet50(pretrained=True),
                'output_size': 2048,
                'output_layer': -1},
            "Alexnet": {
                'model': models.alexnet(pretrained=True),
                'output_size': 9216,
                    'output_layer': -1},
            "VGG16": {
                'model': models.vgg16(pretrained=True),
                'output_size': 25088,
                'output_layer': -1},
        }
        model = available_models["encoder_name"]['model']
        model_layers = list(model.children())
        encoder_layers = available_models[model_name]['output_layer']
        encoder = nn.Sequential(*model_layers[:encoder_layers])
        self.decoder_size = available_models[model_name]["output_size"]
        return encoder

    def forward(self, x):
        batch_size, timesteps = x.size()[0], x.size()[1]
        state = self._initialize_state(b_size=batch_size)

        feature_maps = []
        with torch.no_grad():
            for t in range(timesteps):
                feature_map = self.encoder(x[:, t, :, :, :])
                feature_map = feature_map.view(batch_size, -1)
                feature_maps.append(feature_map)
        convs = torch.stack(feature_maps, 0)
        logit = self.decoder(convs, state)
        return logit

    def _initialize_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            V(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01)),
            V(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01))
        )