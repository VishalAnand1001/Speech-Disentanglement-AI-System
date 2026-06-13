# Speech Disentanglement AI System

An AI-powered speech analysis system designed to isolate a target speaker from a mixed audio recording and detect user-specified keywords within the extracted speech. The system combines noise suppression, speaker recognition, speech separation, and keyword spotting into a unified pipeline with a modern web interface.

## Authors

- Vishal Anand
- Archit Dhiman

---

## Overview

In real-world audio recordings, multiple speakers often overlap, making it difficult to analyze the speech of a specific individual. This project addresses that challenge by allowing users to upload a sample recording of a target speaker and a mixed recording containing multiple speakers. The system then automatically identifies and isolates the target speaker and searches for a user-defined keyword within the extracted speech.

The project is particularly useful in applications such as scam call analysis, forensic investigations, surveillance systems, meeting analysis, and customer support monitoring.

---

## Key Features

### Noise Suppression
- Removes environmental noise and enhances audio quality using DeepFilterNet.
- Improves the performance of subsequent speech processing stages.

### Speaker Enrollment
- Creates a unique voiceprint of the target speaker.
- Uses ECAPA-TDNN speaker embeddings for speaker recognition.

### Speaker Isolation
- Separates mixed speech into individual speakers using SepFormer.
- Identifies the target speaker through voiceprint similarity matching.

### Keyword Detection
- Converts isolated speech into text using Faster-Whisper.
- Detects custom user-defined keywords.
- Displays transcript and matching timestamps.

### Web Interface
- Interactive React frontend.
- Flask backend API.
- Audio playback support.
- Real-time processing feedback and results display.

---

## System Architecture

Target Speaker Audio
↓
Noise Suppression
↓
Voiceprint Creation
↓
Target Speaker Embedding

Mixed Audio
↓
Noise Suppression
↓
Speaker Separation
↓
Speaker Matching
↓
Target Speaker Isolation
↓
Speech-to-Text Conversion
↓
Keyword Detection
↓
Results Dashboard

---

## Technology Stack

### Frontend
- React
- JavaScript
- CSS

### Backend
- Flask
- Flask-CORS

### Artificial Intelligence & Machine Learning
- PyTorch
- SpeechBrain
- ECAPA-TDNN
- SepFormer
- Faster-Whisper
- DeepFilterNet

### Audio Processing
- Torchaudio
- FFmpeg

---

## Project Structure

Speech-Disentanglement-AI-System/

├── frontend/

│   ├── src/

│   ├── public/

│   └── package.json

│

├── stage1_listen/

│   └── noise_suppression.py

│

├── stage2_focus/

│   ├── enrollment.py

│   └── speaker_isolation.py

│

├── stage3_detect/

│   └── keyword_spotting.py

│

├── data/

│   ├── 01_raw_noisy/

│   ├── 03_cleaned_audio/

│   └── 04_isolated_audio/

│

├── models/

│

├── app.py

├── main.py

├── requirements.txt

└── README.md

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Speech-Disentanglement-AI-System.git

cd Speech-Disentanglement-AI-System
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Backend

```bash
python app.py
```

The backend server will start on:

```text
http://localhost:5000
```

---

## Running the Frontend

Navigate to the frontend folder:

```bash
cd frontend
```

Install frontend dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

The frontend will start on:

```text
http://localhost:5173
```

---

## Usage

1. Open the web application.
2. Upload an enrollment audio sample of the target speaker.
3. Upload the mixed audio recording containing multiple speakers.
4. Enter the keyword to search.
5. Click the Analyze button.
6. The system will:
   - Remove background noise.
   - Generate a voiceprint of the target speaker.
   - Separate speakers from the mixed recording.
   - Isolate the target speaker.
   - Convert speech to text.
   - Detect the specified keyword.
7. View the isolated audio, transcript, and keyword matches through the results dashboard.

---

## Applications

- Scam Call Analysis
- Audio Forensics
- Surveillance Systems
- Criminal Investigations
- Customer Support Monitoring
- Meeting Analytics
- Security and Intelligence Operations
- Multi-Speaker Audio Processing

---

## Future Enhancements

- Real-time audio processing
- Multi-keyword search
- Speaker diarization
- Confidence score visualization
- Multi-language support
- Cloud deployment
- Live microphone integration
- Enhanced speaker verification

---

## License

This project is developed for educational, research, and academic purposes.

---

## Acknowledgements

This project builds upon several open-source technologies and research contributions including:

- PyTorch
- SpeechBrain
- DeepFilterNet
- Faster-Whisper
- SepFormer

The authors would like to thank the open-source community for providing the tools, models, and resources that made this project possible.
