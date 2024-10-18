# Whisper Transcription Azure Web App

This is a web application that uses OpenAI's Whisper model to transcribe audio files. It's built with Flask and deployed on Azure App Service.

## Features

- Upload audio files
- Transcribe audio using Whisper large-v3-turbo model
- Download transcription results

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Azure Storage and update the connection string
4. Run locally: `python app.py`

## Deployment

This app is designed to be deployed on Azure App Service. See Azure documentation for deployment instructions.