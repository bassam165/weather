import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# INFORMER MODEL
class Informer(torch.nn.Module):
    def __init__(self, input_size, output_size, input_window, output_window,
                 d_model=128, nhead=8, num_layers=3, dropout=0.3):
        super(Informer, self).__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.d_model = d_model

        # Project input features to d_model
        self.input_projection = torch.nn.Linear(input_size, d_model)
        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final FC layer
        self.fc = torch.nn.Linear(input_window * d_model, output_window * output_size)

    def forward(self, x):
        # x shape: (batch_size, input_window, input_size)
        batch_size = x.size(0)
        x = self.input_projection(x)                
        x = x.permute(1, 0, 2)                     
        enc_out = self.encoder(x)                   
        enc_out = enc_out.permute(1, 0, 2)        
        enc_out = enc_out.contiguous().view(batch_size, -1)  
        out = self.fc(enc_out)                      
        out = out.view(batch_size, self.output_window, -1)   
        return out


# MODEL PARAMS
input_size = 15         # Number of features
output_size = 6         # Number of targets
INPUT_WINDOW = 30       # Input sequence length
OUTPUT_WINDOW = 30      # Output (forecast) length

# LOAD MODEL

model = Informer(
    input_size=input_size,
    output_size=output_size,
    input_window=INPUT_WINDOW,
    output_window=OUTPUT_WINDOW
)
checkpoint_path = "best_informer_model.pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
except Exception as e:
    # Model load error displayed in Streamlit
    st.error(f"Failed to load model: {e}")


# DATASET DEFINITION
class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, feature_cols: list, input_window: int):
        self.features = data[feature_cols].values
        self.input_window = input_window

    def __len__(self):
        # Number of available windows
        return max(0, len(self.features) - self.input_window + 1)

    def __getitem__(self, idx: int):
        # Return the slice of length input_window
        if idx + self.input_window > len(self.features):
            raise IndexError("Index out of range for dataset.")
        x = self.features[idx : idx + self.input_window]
        # x shape: (input_window, num_features)
        return torch.tensor(x, dtype=torch.float32)


# STREAMLIT APP
st.title("Disaster Prediction Rajshahi")

option = st.selectbox("Select Input Type", ("Manual Entry", "Upload CSV"))

feature_cols = [
    'precipitation_sum', 'rain_sum', 'precipitation_hours',
    'sealevelpressure', 'humidity', 'river_discharge', 'cloudcover',
    'Magnitude', 'windspeed_10m_max', 'winddir', 'temperature_2m_mean',
    'solarenergy', 'et0_fao_evapotranspiration',
    'temperature_2m_max', 'apparent_temperature_max(Â°C)'
]

target_cols = [
    'flood_target', 'earthquake_target', 'cyclone_target',
    'water_scarcity_target', 'wildfire_target', 'extreme_heat_target'
]

#  MANUAL ENTRY
if option == "Manual Entry":
    st.write("Enter feature values for a single day:")
    # Gather a single row of data from the user
    input_data = [st.number_input(col, value=0.0) for col in feature_cols]

    if st.button("Predict"):


        row_tensor = torch.tensor(input_data, dtype=torch.float32)
        # 2) expand to shape (30, 15) by repeating row 30 times
        row_30 = row_tensor.unsqueeze(0).repeat(INPUT_WINDOW, 1)
        # 3) add batch dimension -> shape (1, 30, 15)
        input_tensor = row_30.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            prediction = model(input_tensor)

        st.write("\n**Predicted Values (Day 30)**")
        for i, target in enumerate(target_cols):
            st.write(f"{target}: {prediction[0, -1, i].item():.4f}")

#  UPLOAD CSV
elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Check columns
        if not all(col in df.columns for col in feature_cols):
            st.error("Invalid CSV file. Ensure it contains all required columns.")
        else:
            # 1) Filter to feature columns
            df = df[feature_cols]
            # 2) Scale features
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=feature_cols)

            # Must have at least 30 rows
            if len(df_scaled) < INPUT_WINDOW:
                st.error("CSV must contain at least 30 rows of data for a 30-day window.")
            else:
                # Build dataset
                dataset = TimeSeriesDataset(df_scaled, feature_cols, INPUT_WINDOW)
                # Grab the last window
                last_idx = len(dataset) - 1
                input_window_data = dataset[last_idx]  # shape (30, 15)

                # Add batch dimension => shape (1, 30, 15)
                input_tensor = input_window_data.unsqueeze(0)

                # Run model
                with torch.no_grad():
                    predictions = model(input_tensor)

                # predictions shape: (1, 30, 6) => 30 future time steps for 6 targets
                st.write("\n**Predicted Values for Next 30 Days**")
                pred_array = predictions.squeeze(0).numpy()
                pred_df = pd.DataFrame(pred_array, columns=target_cols)
                st.dataframe(pred_df)

                st.success("Prediction complete!")
