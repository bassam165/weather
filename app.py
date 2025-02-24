import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# Load trained model
class Informer(torch.nn.Module):
    def __init__(self, input_size, output_size, input_window, output_window,
                 d_model=128, nhead=8, num_layers=3, dropout=0.3):
        super(Informer, self).__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.d_model = d_model

        self.input_projection = torch.nn.Linear(input_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(input_window * d_model, output_window * output_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        enc_out = self.encoder(x)
        enc_out = enc_out.permute(1, 0, 2)
        enc_out = enc_out.contiguous().view(batch_size, -1)
        out = self.fc(enc_out)
        out = out.view(batch_size, self.output_window, -1)
        return out

# Load model
input_size = 15  # Adjusted input size to match saved model
output_size = 6   # Adjust based on actual target variables
INPUT_WINDOW = 30
OUTPUT_WINDOW = 30

model = Informer(input_size, output_size, INPUT_WINDOW, OUTPUT_WINDOW)

# Handle size mismatch error
checkpoint = torch.load("best_informer_model.pth", map_location=torch.device('cpu'))
checkpoint_keys = checkpoint.keys()

if "input_projection.weight" in checkpoint_keys and checkpoint["input_projection.weight"].shape[1] != input_size:
    model.input_projection = torch.nn.Linear(checkpoint["input_projection.weight"].shape[1], model.d_model)
    model.fc = torch.nn.Linear(INPUT_WINDOW * model.d_model, OUTPUT_WINDOW * output_size)

model.load_state_dict(checkpoint, strict=False)
model.eval()

# Define dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, feature_cols, input_window):
        self.features = data[feature_cols].values
        self.input_window = input_window

    def __len__(self):
        return len(self.features) - self.input_window

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.input_window]
        return torch.tensor(x, dtype=torch.float32)

# Streamlit App
st.title("Disaster Prediction Rajshahi")
option = st.selectbox("Select Input Type", ("Manual Entry", "Upload CSV"))

feature_cols = [
    'precipitation_sum', 'rain_sum', 'precipitation_hours',
    'sealevelpressure', 'humidity', 'river_discharge', 'cloudcover',
    'Magnitude', 'windspeed_10m_max', 'winddir', 'temperature_2m_mean',
    'solarenergy', 'et0_fao_evapotranspiration',
    'temperature_2m_max', 'apparent_temperature_max(Â°C)'
]

target_cols = ['flood_target', 'earthquake_target', 'cyclone_target',
               'water_scarcity_target', 'wildfire_target', 'extreme_heat_target']

if option == "Manual Entry":
    st.write("Enter feature values for prediction")
    input_data = []
    for col in feature_cols:
        value = st.number_input(col, value=0.0)
        input_data.append(value)
    
    if st.button("Predict"):
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.expand(-1, INPUT_WINDOW, -1)
        with torch.no_grad():
            prediction = model(input_tensor)
        st.write("Predicted Values:")
        for i, target in enumerate(target_cols):
            st.write(f"{target}: {prediction[0, -1, i].item():.4f}")

elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in feature_cols):
            df = df[feature_cols]
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=feature_cols)
            dataset = TimeSeriesDataset(df_scaled, feature_cols, INPUT_WINDOW)
            
            input_tensor = torch.tensor(dataset[-1], dtype=torch.float32).unsqueeze(0)
            input_tensor = input_tensor.expand(-1, INPUT_WINDOW, -1)
            with torch.no_grad():
                predictions = model(input_tensor)
            
            st.write("Predicted Values for Next 30 Days:")
            pred_df = pd.DataFrame(predictions.squeeze(0).numpy(), columns=target_cols)
            st.dataframe(pred_df)
        else:
            st.write("Invalid CSV file. Ensure it contains the required columns.")