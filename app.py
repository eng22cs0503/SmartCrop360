from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
import io
import json
import re
import os
import sys
import sqlite3
import secrets
from difflib import get_close_matches
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load .env from backend/croprag/
load_dotenv(Path(__file__).parent.parent / ".env")

# -------------------------
# Auth setup — bcrypt + JWT
# -------------------------
from passlib.context import CryptContext
from jose import JWTError, jwt

_JWT_SECRET  = os.getenv("JWT_SECRET", secrets.token_hex(32))
_JWT_ALG     = "HS256"
_JWT_EXPIRE  = 60 * 24 * 7   # 7 days in minutes
_pwd_ctx     = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer      = HTTPBearer(auto_error=False)

# SQLite user store — one file next to the .env
_DB_PATH = Path(__file__).parent.parent / "users.db"

def _get_db():
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username    TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                email       TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                region      TEXT DEFAULT 'South India',
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

_init_db()

def _create_token(username: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=_JWT_EXPIRE)
    return jwt.encode({"sub": username, "exp": exp}, _JWT_SECRET, algorithm=_JWT_ALG)

def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, _JWT_SECRET, algorithms=[_JWT_ALG])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

from .rag.image_predictor import ImagePredictor
from .schema.prediction import PredictionResponse

# -------------------------
# RAG pipeline + Mistral — loaded once at startup
# -------------------------
rag_pipeline = None
mistral_client = None

try:
    _pkg_root = str(Path(__file__).parent.parent)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)

    from croprag_fastapi.rag.pipeline import RAGPipeline
    from llm.mistral_client import MistralClientWrapper

    rag_pipeline   = RAGPipeline()
    mistral_client = MistralClientWrapper()
    print("✅ RAG pipeline and Mistral client loaded")
except Exception as _e:
    print(f"⚠️  RAG/Mistral unavailable: {_e}")

app = FastAPI(title="Crop Disease Detection API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Initialize ML predictor
# -------------------------
predictor = ImagePredictor()

# -------------------------
# Load disease knowledge base
# -------------------------
json_path = Path(__file__).parent.parent / "dataset" / "crop_disease_rag_10_plants_full.json"

with open(json_path, "r", encoding="utf-8") as f:
    disease_data = json.load(f)

# Normalize disease names
disease_map = {}
for item in disease_data:
    key = item["disease"].lower()
    key = key.replace("_", " ")
    key = re.sub(r"[^\w\s]", "", key).strip()
    disease_map[key] = item


def normalize_label(text: str) -> str:
    text = text.lower()
    text = text.replace("___", " ").replace("__", " ").replace("_", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def lookup_disease(label: str):
    label_clean = normalize_label(label)

    # Healthy plant — no disease
    if "healthy" in label_clean.split():
        return None

    # 1. Exact / fuzzy match on disease name — must score well
    match = get_close_matches(label_clean, disease_map.keys(), n=1, cutoff=0.6)
    if match:
        return disease_map[match[0]]

    # 2. Try lower cutoff but require the crop word to also be present
    match = get_close_matches(label_clean, disease_map.keys(), n=3, cutoff=0.4)
    if match:
        # Pick the match that shares the most words with the label
        label_words = set(label_clean.split())
        best = max(match, key=lambda m: len(set(m.split()) & label_words))
        return disease_map[best]

    # 3. Word-overlap scoring — every word in label must contribute
    label_words = set(label_clean.split())
    best, best_score = None, 0
    for key, item in disease_map.items():
        key_words = set(key.split())
        score = len(label_words & key_words)
        if score > best_score:
            best_score = score
            best = item

    if best_score >= 2:  # require at least 2 matching words (e.g. "tomato" + "blight")
        return best

    # 4. Symptom keyword match as last resort
    best, best_score = None, 0
    for item in disease_data:
        symptoms_text = " ".join(item.get("symptoms", [])).lower()
        score = sum(1 for w in label_words if len(w) > 3 and w in symptoms_text)
        if score > best_score:
            best_score = score
            best = item

    return best if best_score > 0 else None


# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "Backend running"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "API is running"}


# ==========================================================
# IMAGE PREDICTION
# ==========================================================
@app.post("/api/v1/disease/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if rag_pipeline is None or mistral_client is None:
        return {"disease": "Error", "label": "error", "confidence": 0.0, "score": 0.0,
                "severity": "Unknown", "recommendation": "RAG/Mistral pipeline not available. Check MISTRAL_API_KEY.", "crop": None}
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        label, confidence = predictor.predict(image)

        if "background" in label.lower():
            return {"disease": "Background", "label": label, "confidence": float(confidence),
                    "score": float(confidence), "severity": "Low",
                    "recommendation": "No plant leaf detected in the image.", "crop": None}

        if "healthy" in label.lower():
            crop_name = normalize_label(label).replace("healthy", "").strip().title()
            return {"disease": "Healthy", "label": label, "confidence": float(confidence),
                    "score": float(confidence), "severity": "None",
                    "recommendation": f"No disease detected. {crop_name} plant appears healthy. Continue regular monitoring and preventive care.",
                    "crop": crop_name or None}

        disease_info = lookup_disease(label)
        crop_name    = disease_info.get("crop", "") if disease_info else ""
        disease_name = disease_info.get("disease", label) if disease_info else label
        severity     = "High" if confidence > 80 else "Moderate" if confidence > 60 else "Low"

        # CLIP image embedding → ChromaDB retrieval → Mistral generation
        context_docs   = rag_pipeline.query_by_image(image, k=3)
        recommendation = mistral_client.generate(crop_name, disease_name, context_docs)

        return {
            "disease":        disease_name.replace("_", " ").title(),
            "label":          label,
            "confidence":     float(confidence),
            "score":          float(confidence),
            "severity":       severity,
            "recommendation": recommendation,
            "crop":           crop_name,
            "soil_ph":        disease_info.get("soil_ph") if disease_info else None,
        }
    except Exception as e:
        return {"disease": "Error", "label": "error", "confidence": 0.0, "score": 0.0,
                "severity": "Unknown", "recommendation": f"Error: {str(e)}", "crop": None}


# ==========================================================
# TEXT PREDICTION
# ==========================================================
@app.post("/api/v1/disease/predict/text")
def text_query(body: dict = Body(...)):
    if rag_pipeline is None or mistral_client is None:
        return {"disease": "Error", "label": "error", "confidence": 0.0, "score": 0.0,
                "severity": "Unknown", "recommendation": "RAG/Mistral pipeline not available. Check MISTRAL_API_KEY.", "crop": None}
    try:
        query = body.get("query", "") or body.get("symptoms", "")
        if not query:
            return {"disease": "No Query", "label": "no_query", "confidence": 0.0, "score": 0.0,
                    "severity": "Unknown", "recommendation": "Please provide a disease name or symptoms.", "crop": None}

        disease_info = lookup_disease(query)
        crop_name    = disease_info.get("crop", "") if disease_info else ""
        disease_name = disease_info.get("disease", query) if disease_info else query

        # CLIP text embedding → ChromaDB retrieval → Mistral generation
        context_docs   = rag_pipeline.query_text(crop_name, disease_name, k=3)
        recommendation = mistral_client.generate(crop_name, disease_name, context_docs)

        return {
            "disease":        disease_name.replace("_", " ").title(),
            "label":          query,
            "confidence":     75.0,
            "score":          75.0,
            "severity":       "Moderate",
            "recommendation": recommendation,
            "crop":           crop_name,
            "soil_ph":        disease_info.get("soil_ph") if disease_info else None,
        }
    except Exception as e:
        return {"disease": "Error", "label": "error", "confidence": 0.0, "score": 0.0,
                "severity": "Unknown", "recommendation": f"Error: {str(e)}", "crop": None}


# ==========================================================
# CHAT
# ==========================================================
@app.post("/api/v1/chat")
def chat(body: dict = Body(...)):
    if rag_pipeline is None or mistral_client is None:
        msg = "RAG/Mistral pipeline not available. Check MISTRAL_API_KEY."
        return {"reply": msg, "response": msg, "message": msg}
    try:
        message = body.get("message", "").strip()
        crop    = (body.get("crop",    "") or "").strip().lower()
        disease = (body.get("disease", "") or "").strip().lower()
        history = body.get("history", [])  # optional conversation history

        if not message:
            return {"reply": "Please type a message.", "response": "", "message": ""}

        # Resolve crop/disease context to a specific disease entry if available
        if crop or disease:
            disease_info = lookup_disease(disease) if disease else None
            if not disease_info and crop:
                matches = [d for d in disease_data if d.get("crop", "").lower() == crop]
                disease_info = matches[0] if matches else None
            if disease_info:
                crop    = disease_info.get("crop", crop)
                disease = disease_info.get("disease", disease)

        # CLIP text embedding → ChromaDB retrieval → Mistral chat
        context_docs = rag_pipeline.query_text(crop, f"{disease} {message}".strip(), k=3)
        reply        = mistral_client.chat(message, crop, disease, context_docs, history)
        return {"reply": reply, "response": reply, "message": reply}
    except Exception as e:
        msg = f"Error generating response: {str(e)}"
        return {"reply": msg, "response": msg, "message": msg}


# ==========================================================
# AUTH ENDPOINTS
# ==========================================================
@app.post("/api/v1/auth/register")
def register(body: dict = Body(...)):
    username = (body.get("username") or "").strip()
    name     = (body.get("name") or body.get("full_name") or username).strip()
    email    = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not username or not email or not password:
        raise HTTPException(status_code=400, detail="username, email and password are required")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    pw_hash = _pwd_ctx.hash(password)
    try:
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, name, email, password_hash) VALUES (?,?,?,?)",
                (username, name, email, pw_hash)
            )
            conn.commit()
    except sqlite3.IntegrityError as e:
        msg = "Email already registered" if "email" in str(e) else "Username already taken"
        raise HTTPException(status_code=409, detail=msg)

    token = _create_token(username)
    return {"success": True, "access_token": token, "token_type": "bearer",
            "user": {"username": username, "name": name, "email": email, "region": "South India"}}


@app.post("/api/v1/auth/login")
def login(body: dict = Body(...)):
    identifier = (body.get("username") or body.get("email") or "").strip()
    password   = body.get("password") or ""

    if not identifier or not password:
        raise HTTPException(status_code=400, detail="username/email and password are required")

    with _get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? OR email=?", (identifier, identifier.lower())
        ).fetchone()

    if not row or not _pwd_ctx.verify(password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username/email or password")

    token = _create_token(row["username"])
    return {"success": True, "access_token": token, "token_type": "bearer",
            "user": {"username": row["username"], "name": row["name"],
                     "email": row["email"], "region": row["region"] or "South India"}}


@app.get("/api/v1/auth/me")
def get_me(username: str = Depends(_verify_token)):
    with _get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": row["username"], "name": row["name"],
            "email": row["email"], "region": row["region"]}


@app.patch("/api/v1/auth/me")
def update_me(body: dict = Body(...), username: str = Depends(_verify_token)):
    region = body.get("region")
    name   = body.get("name")
    if region:
        with _get_db() as conn:
            conn.execute("UPDATE users SET region=? WHERE username=?", (region, username))
            conn.commit()
    if name:
        with _get_db() as conn:
            conn.execute("UPDATE users SET name=? WHERE username=?", (name, username))
            conn.commit()
    return {"success": True}


# ==========================================================
# SERVE FRONTEND
# ==========================================================
@app.get("/app")
def serve_frontend():
    frontend_path = Path(__file__).parent.parent.parent.parent / "smartcrop360.html"
    return FileResponse(frontend_path)


# ==========================================================
# DASHBOARD & FIELD DATA
# ==========================================================
@app.get("/api/v1/dashboard/stats")
def get_dashboard_stats():
    """Get real-time field statistics"""
    return {
        "soil_moisture": 23,
        "temperature": 28,
        "humidity": 67,
        "last_updated": "2024-03-13T10:30:00Z"
    }


@app.get("/api/v1/fields")
def get_fields():
    """Get all field information"""
    return {
        "fields": [
            {
                "id": 1,
                "name": "Field 1 — Wheat",
                "crop": "🌾",
                "lat": 20.5937,
                "lng": 78.9629,
                "severity": "Low",
                "color": "#6ecb63",
                "area": "5 hectares",
                "health_score": 92
            },
            {
                "id": 2,
                "name": "Field 2 — Tomato",
                "crop": "🍅",
                "lat": 21.1458,
                "lng": 79.0882,
                "severity": "High",
                "color": "#e05555",
                "area": "3 hectares",
                "health_score": 45
            },
            {
                "id": 3,
                "name": "Field 3 — Potato",
                "crop": "🥔",
                "lat": 19.7515,
                "lng": 75.7139,
                "severity": "Medium",
                "color": "#c9a84c",
                "area": "4 hectares",
                "health_score": 68
            }
        ]
    }


# ==========================================================
# WEATHER API
# ==========================================================
@app.get("/api/v1/weather/forecast")
def get_weather_forecast():
    """Get 5-day weather forecast"""
    return {
        "forecast": [
            {"day": "Today", "icon": "☀️", "temp": 28, "desc": "Clear", "humidity": 45, "wind": 12},
            {"day": "Tue", "icon": "🌦️", "temp": 24, "desc": "Heavy Rain", "humidity": 85, "wind": 25},
            {"day": "Wed", "icon": "⛅", "temp": 26, "desc": "Partly Cloudy", "humidity": 60, "wind": 15},
            {"day": "Thu", "icon": "🌤️", "temp": 30, "desc": "Sunny", "humidity": 40, "wind": 10},
            {"day": "Fri", "icon": "🌧️", "temp": 22, "desc": "Rain", "humidity": 80, "wind": 20}
        ],
        "alerts": [
            {
                "type": "warning",
                "title": "Heavy Rainfall Alert",
                "message": "Expected 80–120mm over 24 hours starting tomorrow. Ensure proper field drainage. Avoid pesticide application 12 hrs before rain.",
                "severity": "high"
            }
        ]
    }


# ==========================================================
# ANALYTICS API - DYNAMIC BASED ON CROP
# ==========================================================
@app.get("/api/v1/analytics/rainfall")
def get_rainfall_data(period: str = "6m", crop: str = None):
    """Get rainfall analytics data with RAG-powered crop-specific insights"""

    data_map = {
        "6m": {"labels": ["Jan","Feb","Mar","Apr","May","Jun"], "rainfall": [50,70,40,60,90,30], "moisture": [22,28,18,24,32,19]},
        "3m": {"labels": ["Apr","May","Jun"], "rainfall": [60,90,30], "moisture": [24,32,19]},
        "1y": {"labels": ["J","F","M","A","M","J","J","A","S","O","N","D"], "rainfall": [40,55,70,45,60,80,40,55,90,30,45,65], "moisture": [20,22,28,18,24,32,19,22,28,21,18,20]}
    }

    result = dict(data_map.get(period, data_map["6m"]))

    if crop:
        crop_diseases = [d for d in disease_data if d.get("crop", "").lower() == crop.lower()]
        if crop_diseases:
            result["crop"] = crop_diseases[0].get("crop")
            result["optimal_ph"] = crop_diseases[0].get("soil_ph")
            result["common_diseases"] = len(crop_diseases)
            # RAG insight for this crop's moisture/rainfall sensitivity
            try:
                docs    = rag_pipeline.query_text(crop, "irrigation rainfall sensitivity") if rag_pipeline else []
                insight = mistral_client.generate(crop, "irrigation and rainfall sensitivity", docs) if mistral_client else None
                if insight:
                    result["insight"] = insight
            except Exception:
                pass

    return result


@app.get("/api/v1/analytics/disease-trend")
def get_disease_trend(crop: str = None, disease: str = None):
    """Get disease trend with RAG-powered analysis"""
    import random

    if crop or disease:
        random.seed(hash(crop or disease) % 1000)
        labels = ["Week 1","Week 2","Week 3","Week 4"]
        data = [random.randint(5, 20) for _ in range(4)]
        title = f"{crop.title() if crop else disease.replace('_',' ').title()} Disease Trend"

        # RAG insight for this crop/disease trend
        try:
            q    = disease or f"{crop} disease spread"
            docs = rag_pipeline.query_text(crop or "", q) if rag_pipeline else []
            insight = mistral_client.generate(crop or "", q, docs) if mistral_client else None
        except Exception:
            insight = None
        return {"labels": labels, "data": data, "title": title, "crop": crop, "disease": disease, "insight": insight}

    return {"labels": ["Week 1","Week 2","Week 3","Week 4"], "data": [12,19,8,15], "title": "Overall Disease Trend"}


@app.get("/api/v1/analytics/insights")
def get_analytics_insights(crop: str = None, disease: str = None, severity: str = None):
    """RAG-powered analytics insights — specific to the diagnosed crop + disease state"""
    crop    = (crop    or "").strip()
    disease = (disease or "").strip()
    severity = (severity or "").strip()

    is_healthy = "healthy" in disease.lower() or severity.lower() == "none"

    if is_healthy:
        query = f"{crop} healthy plant care preventive maintenance monitoring"
    else:
        sev_context = f"severity: {severity}" if severity else ""
        query = f"{crop} {disease} {sev_context} treatment management symptoms field impact".strip()

    try:
        docs    = rag_pipeline.query_text(crop, query) if rag_pipeline else []
        insight = mistral_client.generate(crop, query, docs) if mistral_client else None
    except Exception:
        insight = None

    if not insight:
        if is_healthy:
            insight = f"{crop} appears healthy. Continue regular monitoring, maintain proper irrigation, and apply preventive fungicide sprays during high-humidity periods."
        elif crop and disease:
            crop_diseases = [d for d in disease_data if d.get("crop", "").lower() == crop.lower()]
            match = next((d for d in crop_diseases if disease.lower() in d.get("disease","").lower()), None)
            if match:
                insight = f"{disease} on {crop}: {match.get('remedy', '')}. Prevention: {'; '.join(match.get('prevention', [])[:2])}."
            else:
                insight = f"Run a disease detection first to get specific AI-powered insights for {crop}."
        else:
            insight = "Run a disease detection first to get AI-powered field insights."

    return {"insight": insight, "crop": crop, "disease": disease, "severity": severity}


# ==========================================================
# ALERTS API
# ==========================================================
@app.get("/api/v1/alerts")
def get_alerts():
    """Get all active alerts"""
    return {
        "alerts": [
            {
                "id": 1,
                "type": "critical",
                "title": "Disease Detected",
                "message": "Tomato Leaf Blight in Field 2",
                "time": "2 hours ago",
                "field": "Field 2",
                "severity": "high"
            },
            {
                "id": 2,
                "type": "warning",
                "title": "Weather Warning",
                "message": "Heavy rainfall expected in 24 hours",
                "time": "5 hours ago",
                "severity": "medium"
            },
            {
                "id": 3,
                "type": "info",
                "title": "Maintenance",
                "message": "Irrigation system requires checkup",
                "time": "1 day ago",
                "severity": "low"
            }
        ],
        "count": 3
    }


@app.delete("/api/v1/alerts/{alert_id}")
def dismiss_alert(alert_id: int):
    """Dismiss an alert"""
    return {"success": True, "message": f"Alert {alert_id} dismissed"}


# ==========================================================
# RECOMMENDATIONS API - DYNAMIC BASED ON CROP/DISEASE
# ==========================================================
@app.get("/api/v1/recommendations")
def get_recommendations(category: str = "all", crop: str = None, disease: str = None):
    """Get farming recommendations by category, optionally filtered by crop/disease"""
    
    # If crop or disease specified, provide targeted recommendations
    if crop or disease:
        disease_info = None
        if disease:
            disease_info = lookup_disease(disease)
        elif crop:
            # Find diseases for this crop
            crop_diseases = [d for d in disease_data if d.get("crop", "").lower() == crop.lower()]
            if crop_diseases:
                disease_info = crop_diseases[0]
        
        if disease_info:
            crop_name = disease_info.get("crop", "your crop")
            disease_name = disease_info.get("disease", "").replace("_", " ").title()
            remedy = disease_info.get("remedy", "")
            prevention_list = disease_info.get("prevention", [])
            symptoms = disease_info.get("symptoms", [])
            soil_ph = disease_info.get("soil_ph", "")
            
            # Build dynamic recommendations
            dynamic_recs = {
                "preventive": [],
                "treatment": [],
                "eco": [],
                "irrigation": []
            }
            
            # Preventive measures from disease data
            for i, prev in enumerate(prevention_list[:4]):
                dynamic_recs["preventive"].append({
                    "icon": ["shield-alt", "seedling", "broom", "eye"][i % 4],
                    "text": f"{crop_name}: {prev}"
                })
            
            # Treatment recommendations
            if remedy:
                dynamic_recs["treatment"].append({
                    "icon": "syringe",
                    "text": f"{crop_name} - {disease_name}: {remedy}"
                })
            
            # Add symptom monitoring
            if symptoms:
                dynamic_recs["treatment"].append({
                    "icon": "search",
                    "text": f"Monitor for: {', '.join(symptoms[:2])}"
                })
            
            # Soil pH recommendation
            if soil_ph:
                dynamic_recs["eco"].append({
                    "icon": "flask",
                    "text": f"{crop_name} thrives in soil pH {soil_ph}. Test and adjust accordingly."
                })
            
            # Crop-specific eco recommendations
            dynamic_recs["eco"].append({
                "icon": "leaf",
                "text": f"Use organic mulch around {crop_name} to retain moisture and suppress weeds"
            })
            dynamic_recs["eco"].append({
                "icon": "recycle",
                "text": f"Compost {crop_name} plant residues (only if disease-free) to enrich soil"
            })
            
            # Irrigation based on crop
            dynamic_recs["irrigation"].append({
                "icon": "tint",
                "text": f"{crop_name} benefits from consistent moisture. Use drip irrigation to minimize leaf wetness."
            })
            dynamic_recs["irrigation"].append({
                "icon": "clock",
                "text": f"Water {crop_name} early morning to allow foliage to dry before evening"
            })
            
            if category == "all":
                return dynamic_recs
            return {category: dynamic_recs.get(category, [])}
    
    # Default generic recommendations
    recommendations = {
        "preventive": [
            {"icon": "sync-alt", "text": "Crop rotation every 2–3 seasons to reduce soil-borne pathogen buildup"},
            {"icon": "seedling", "text": "Choose disease-resistant varieties certified for your region"},
            {"icon": "broom", "text": "Regular field sanitation — remove and dispose of infected plant debris"},
            {"icon": "cloud-sun-rain", "text": "Monitor weather daily; avoid irrigation before heavy rainfall forecast"},
            {"icon": "eye", "text": "Weekly visual inspections of all fields, especially during humid seasons"},
            {"icon": "flask", "text": "Conduct soil testing every 6 months to detect nutrient imbalances early"}
        ],
        "treatment": [
            {"icon": "spray-can", "text": "Fungicide spray once a week — rotate chemicals to avoid resistance"},
            {"icon": "leaf", "text": "Use crop-specific organic pesticides; check label for correct dilution"},
            {"icon": "weight", "text": "Adjust dosage based on disease severity — over-application harms soil"},
            {"icon": "cut", "text": "Prune infected branches immediately; seal wounds with copper paste"}
        ],
        "eco": [
            {"icon": "oil-can", "text": "Neem oil spray (5ml/L) every 7–10 days — effective against most fungi and pests"},
            {"icon": "bug", "text": "Introduce beneficial insects: ladybugs and parasitic wasps for natural pest control"},
            {"icon": "tree", "text": "Mulching with dry leaves — suppresses weeds and prevents fungal splash-back"},
            {"icon": "recycle", "text": "Compost organic waste into fertilizer — reduces dependency on chemical inputs"}
        ],
        "irrigation": [
            {"icon": "tint", "text": "Drip irrigation preferred — reduces leaf wetness and fungal risk"},
            {"icon": "clock", "text": "Water early morning so foliage dries before evening — reduces disease"},
            {"icon": "chart-line", "text": "Maintain soil moisture at 20–30% for most vegetable crops"},
            {"icon": "battery-half", "text": "Check irrigation lines monthly for blockages and pressure drops"}
        ]
    }
    
    if category == "all":
        return recommendations
    return {category: recommendations.get(category, [])}


# ==========================================================
# DETECTION HISTORY API
# ==========================================================
@app.get("/api/v1/detections/history")
def get_detection_history():
    """Get recent disease detection history"""
    return {
        "history": [
            {"disease": "Leaf Spot — Field 2", "time": "2d ago", "severity": "medium"},
            {"disease": "Healthy — Field 1", "time": "4d ago", "severity": "low"},
            {"disease": "Blight — Field 2", "time": "1w ago", "severity": "high"}
        ]
    }


@app.post("/api/v1/detections/save")
def save_detection(body: dict = Body(...)):
    """Save a disease detection report"""
    disease = body.get("disease", "Unknown")
    field = body.get("field", "Unknown")
    severity = body.get("severity", "Unknown")
    
    return {
        "success": True,
        "message": "Disease report saved to history!",
        "saved": {
            "disease": disease,
            "field": field,
            "severity": severity,
            "timestamp": "Just now"
        }
    }


# ==========================================================
# CROP INFO API - Get detailed crop information
# ==========================================================
@app.get("/api/v1/crops/{crop_name}")
def get_crop_info(crop_name: str):
    """Get detailed information about a specific crop"""
    crop_diseases = [d for d in disease_data if d.get("crop", "").lower() == crop_name.lower()]
    
    if not crop_diseases:
        return {"error": "Crop not found", "crop": crop_name}
    
    # Aggregate crop information
    all_symptoms = []
    all_prevention = []
    diseases_list = []
    
    for disease in crop_diseases:
        diseases_list.append({
            "name": disease.get("disease", "").replace("_", " ").title(),
            "remedy": disease.get("remedy", "")
        })
        all_symptoms.extend(disease.get("symptoms", []))
        all_prevention.extend(disease.get("prevention", []))
    
    # Remove duplicates
    all_symptoms = list(set(all_symptoms))
    all_prevention = list(set(all_prevention))
    
    return {
        "crop": crop_diseases[0].get("crop"),
        "soil_ph": crop_diseases[0].get("soil_ph"),
        "common_diseases": diseases_list,
        "common_symptoms": all_symptoms[:5],
        "prevention_tips": all_prevention[:5],
        "disease_count": len(crop_diseases)
    }


@app.get("/api/v1/crops")
def get_all_crops():
    """Get list of all available crops"""
    crops = list(set([d.get("crop") for d in disease_data if d.get("crop")]))
    crops.sort()
    
    crop_summary = []
    for crop in crops:
        crop_diseases = [d for d in disease_data if d.get("crop") == crop]
        crop_summary.append({
            "name": crop,
            "disease_count": len(crop_diseases),
            "soil_ph": crop_diseases[0].get("soil_ph") if crop_diseases else "Unknown"
        })
    
    return {"crops": crop_summary, "total": len(crops)}

