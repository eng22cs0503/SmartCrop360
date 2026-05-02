# SmartCrop360 - Precision Farming Intelligence

A full-stack crop disease detection and farm management system with AI-powered diagnostics.

## Quick Start

### 1. Start the Backend Server

```bash
# Option 1: Use the startup script
start_server.bat

# Option 2: Manual start
cd backend/croprag
python -m uvicorn croprag_fastapi.app:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

### 2. Open the Frontend

**⚠️ IMPORTANT:** To avoid CORS issues, access the frontend through the FastAPI server:

```
http://localhost:8000/app
```

**DO NOT** open `smartcrop360.html` directly in your browser (this will show "API Offline" due to CORS restrictions).

When accessed correctly, you'll see a green "API Online" indicator in the top-right corner.

## API Endpoints

### Health Check
- **GET** `/api/health` - Check if API is running

### Disease Detection
- **POST** `/api/v1/disease/predict/image` - Upload leaf image for disease detection
  - Body: `multipart/form-data` with `file` field
  - Returns: disease name, confidence, severity, recommendations

- **POST** `/api/v1/disease/predict/text` - Text-based symptom analysis
  - Body: `{"query": "brown spots on leaves"}`
  - Returns: disease name, confidence, severity, recommendations

### AI Chat
- **POST** `/api/v1/chat` - Chat with AI assistant
  - Body: `{"message": "How to prevent blight?"}`
  - Returns: `{"reply": "..."}`

### Authentication
- **POST** `/api/v1/auth/login` - User login
  - Body: `{"username": "user", "password": "pass"}`
  
- **POST** `/api/v1/auth/register` - User registration
  - Body: `{"username": "user", "email": "email@example.com", "password": "pass"}`

## Testing

Open `test_api.html` in your browser to test all API endpoints interactively.

## Features

- 🌿 Disease Detection (Image & Text-based)
- 🤖 AI Chat Assistant
- 📊 Analytics Dashboard
- 🗺️ Field Mapping
- 💡 Smart Recommendations
- 🔔 Alert System
- 🌓 Dark/Light Theme

## Tech Stack

**Backend:**
- FastAPI
- PyTorch & TorchVision
- PIL (Image Processing)
- Uvicorn (ASGI Server)

**Frontend:**
- Vanilla JavaScript
- Leaflet.js (Maps)
- Chart.js (Analytics)
- Font Awesome (Icons)

## Project Structure

```
├── backend/croprag/
│   ├── croprag_fastapi/
│   │   ├── app.py              # Main FastAPI application
│   │   ├── rag/                # ML prediction models
│   │   └── schema/             # Response schemas
│   ├── dataset/                # Disease knowledge base
│   └── requirements.txt        # Python dependencies
├── smartcrop360.html           # Main frontend application
├── test_api.html               # API testing interface
└── start_server.bat            # Quick start script
```
