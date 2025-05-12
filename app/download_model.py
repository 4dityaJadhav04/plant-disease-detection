# app/download_model.py
import gdown
import os

# Google Drive file ID
file_id = "15r8DAGYrdnmfACohPlPMIflXYUzeQB4X"
# Output path where the model will be saved
output_path = "app/trained_model/plant_disease_prediction_model.h5"

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Download from Google Drive
gdown.download(id=file_id, output=output_path, quiet=False)
