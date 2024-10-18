import os
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from azure.storage.blob import BlobServiceClient
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = Flask(__name__)

# Azure Storage settings
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
container_name = 'audio-files'
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Whisper model settings
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('/tmp', filename)
            file.save(file_path)
            
            # Upload to Azure Blob Storage
            blob_client = container_client.get_blob_client(filename)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data)
            
            # Transcribe audio
            result = pipe(file_path)
            transcription = result["text"]
            
            # Save transcription to a text file
            transcription_filename = f"{os.path.splitext(filename)[0]}_transcription.txt"
            transcription_path = os.path.join('/tmp', transcription_filename)
            with open(transcription_path, 'w') as f:
                f.write(transcription)
            
            # Upload transcription to Azure Blob Storage
            transcription_blob_client = container_client.get_blob_client(transcription_filename)
            with open(transcription_path, "rb") as data:
                transcription_blob_client.upload_blob(data)
            
            return send_file(transcription_path, as_attachment=True)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)