"""
Production-Ready FastAPI Backend for Punjab Soil Predictor
with logging, monitoring, caching, rate limiting, and security
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from functools import lru_cache
import time
from collections import defaultdict
import hashlib
import json
import io
from fastapi.responses import StreamingResponse
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

# PDF Generation
from pdf_generator import PDFReportGenerator

# Configuration Management
class Settings(BaseSettings):
    APP_NAME: str = "Punjab Soil Predictor API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    MODEL_DIR: str = "models"
    LOG_LEVEL: str = "INFO"
    MAX_REQUESTS_PER_MINUTE: int = 60
    CACHE_PREDICTIONS: bool = True
    CACHE_TTL_SECONDS: int = 3600
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

# Logging Configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize FastAPI
app = FastAPI(
    title=get_settings().APP_NAME,
    description="Production-grade AI-powered soil property prediction API",
    version=get_settings().APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate Limiting
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.max_requests = get_settings().MAX_REQUESTS_PER_MINUTE
    
    def is_allowed(self, client_ip: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        self.requests[client_ip].append(now)
        return True

rate_limiter = RateLimiter()

async def check_rate_limit(request: Request):
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later."
        )

# Prediction Cache
class PredictionCache:
    def __init__(self):
        self.cache = {}
        self.ttl = get_settings().CACHE_TTL_SECONDS
    
    def _generate_key(self, lat: float, lon: float, depth: float) -> str:
        """Generate cache key from coordinates"""
        data = f"{lat:.6f}_{lon:.6f}_{depth:.2f}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, lat: float, lon: float, depth: float) -> Optional[Dict]:
        """Get cached prediction"""
        if not get_settings().CACHE_PREDICTIONS:
            return None
        
        key = self._generate_key(lat, lon, depth)
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.info(f"Cache hit for key: {key}")
                return cached_data
            else:
                del self.cache[key]
        return None
    
    def set(self, lat: float, lon: float, depth: float, data: Dict):
        """Cache prediction result"""
        if get_settings().CACHE_PREDICTIONS:
            key = self._generate_key(lat, lon, depth)
            self.cache[key] = (data, time.time())
            logger.info(f"Cached prediction for key: {key}")
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        logger.info("Cache cleared")

prediction_cache = PredictionCache()

# Pydantic Models
class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=30.0, le=32.0, description="Latitude in Punjab region")
    longitude: float = Field(..., ge=74.0, le=77.0, description="Longitude in Punjab region")
    depth: float = Field(3.0, ge=0.5, le=30.0, description="Depth in meters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 31.227389,
                "longitude": 75.766764,
                "depth": 3.0
            }
        }

class PropertyPrediction(BaseModel):
    value: Optional[float]
    r2_score: Optional[float]
    confidence: Optional[str]
    unit: Optional[str]
    error: Optional[str] = None

class PredictionResponse(BaseModel):
    success: bool
    timestamp: datetime
    input: Dict[str, float]
    predictions: Dict[str, PropertyPrediction]
    cached: bool = False
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: bool
    models_count: int
    uptime_seconds: float
    cache_size: int

class MetricsResponse(BaseModel):
    total_predictions: int
    cache_hit_rate: float
    average_response_time_ms: float
    models_info: Dict[str, Any]

# Monitoring
class Metrics:
    def __init__(self):
        self.start_time = time.time()
        self.prediction_count = 0
        self.cache_hits = 0
        self.total_requests = 0
        self.response_times = []
    
    def record_prediction(self, cached: bool, response_time: float):
        self.prediction_count += 1
        self.total_requests += 1
        if cached:
            self.cache_hits += 1
        self.response_times.append(response_time)
        
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    def get_metrics(self) -> Dict:
        return {
            "total_predictions": self.prediction_count,
            "cache_hit_rate": (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0,
            "average_response_time_ms": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "uptime_seconds": time.time() - self.start_time
        }

metrics = Metrics()

# Soil Predictor with Error Handling
class SoilPredictor:
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.poly_features = {}
        self.feature_columns = None
        self.target_columns = []
        self.results_summary = {}
        self.loaded = False
        
        # Property units
        self.units = {
            'N-value': 'blows/30cm',
            'Bulk Density': 'g/cm³',
            'Cohesion': 'kPa',
            'Shear angle': 'degrees',
            'Gravel': '%',
            'Sand': '%',
            'Silt & Clay': '%'
        }
        
    def load_models(self, model_dir: str = None):
        """Load pre-trained models with error handling"""
        if model_dir is None:
            model_dir = get_settings().MODEL_DIR
        
        model_path = f'{model_dir}/predictor_state.pkl'
        
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                state = pickle.load(f)
                self.scalers = state['scalers']
                self.models = state['models']
                self.poly_features = state['poly_features']
                self.feature_columns = state['feature_columns']
                self.target_columns = state['target_columns']
                self.results_summary = state['results_summary']
            
            self.loaded = True
            logger.info(f"✓ Loaded {len(self.models)} models successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            return False
    
    def predict(self, latitude: float, longitude: float, depth: float) -> Dict:
        """Make predictions with comprehensive error handling"""
        if not self.loaded:
            raise RuntimeError("Models not loaded")
        
        # Prepare input
        if depth is not None and len(self.feature_columns) == 3:
            input_data = np.array([[latitude, longitude, depth]])
        else:
            input_data = np.array([[latitude, longitude]])
        
        predictions = {}
        
        for target in self.target_columns:
            if target in self.models:
                try:
                    # Apply polynomial features if used
                    if self.poly_features.get(target) is not None:
                        input_transformed = self.poly_features[target].transform(input_data)
                    else:
                        input_transformed = input_data
                    
                    # Scale input
                    input_scaled = self.scalers[target].transform(input_transformed)
                    
                    # Predict
                    prediction = self.models[target].predict(input_scaled)[0]
                    r2_score = self.results_summary[target]['best_r2']
                    
                    confidence = 'High' if r2_score > 0.8 else 'Medium' if r2_score > 0.6 else 'Low'
                    
                    predictions[target] = PropertyPrediction(
                        value=float(prediction),
                        r2_score=float(r2_score),
                        confidence=confidence,
                        unit=self.units.get(target, '')
                    )
                    
                except Exception as e:
                    logger.error(f"Prediction error for {target}: {e}", exc_info=True)
                    predictions[target] = PropertyPrediction(
                        value=None,
                        r2_score=None,
                        confidence=None,
                        unit=None,
                        error=str(e)
                    )
        
        return predictions

def generate_pdf_report(pdf_data):
    """Generate PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("Soil Analysis Report", styles['Title']))
    
    # Location info
    input_data = pdf_data['input']
    story.append(Paragraph(f"Location: {input_data['latitude']:.6f}°N, {input_data['longitude']:.6f}°E", styles['Normal']))
    story.append(Paragraph(f"Depth: {input_data['depth']}m", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Properties table
    props = pdf_data['predictions']
    data = [['Property', 'Value', 'Confidence']]
    for name, pred in props.items():
        data.append([name, f"{pred['value']:.2f}", pred['confidence']])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    
    doc.build(story)
    return buffer.getvalue()




# Initialize predictor
predictor = SoilPredictor()

# Initialize PDF generator
pdf_generator = PDFReportGenerator()

# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("="*70)
    logger.info(f"Starting {get_settings().APP_NAME} v{get_settings().APP_VERSION}")
    logger.info("="*70)
    
    if predictor.load_models():
        logger.info("✓ Application ready")
    else:
        logger.warning("⚠ Application started but models not loaded")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")
    logger.info(f"Total predictions served: {metrics.prediction_count}")

# Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if get_settings().DEBUG else "An error occurred"
        }
    )

# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": get_settings().APP_NAME,
        "version": get_settings().APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
        "metrics": "/api/metrics"
    }

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check with detailed status"""
    models_loaded = predictor.loaded
    uptime = time.time() - metrics.start_time
    
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        timestamp=datetime.now(),
        models_loaded=models_loaded,
        models_count=len(predictor.models),
        uptime_seconds=uptime,
        cache_size=len(prediction_cache.cache)
    )

@app.post("/api/predict", response_model=PredictionResponse, 
          tags=["Prediction"], dependencies=[Depends(check_rate_limit)])
async def predict(request: PredictionRequest):
    """
    Predict soil properties with caching and monitoring
    """
    start_time = time.time()
    
    if not predictor.loaded:
        logger.error("Prediction attempted with no models loaded")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable. Models not loaded."
        )
    
    # Check cache first
    cached_result = prediction_cache.get(
        request.latitude, 
        request.longitude, 
        request.depth
    )
    
    if cached_result:
        processing_time = (time.time() - start_time) * 1000
        metrics.record_prediction(cached=True, response_time=processing_time)
        
        return PredictionResponse(
            success=True,
            timestamp=datetime.now(),
            input={
                "latitude": request.latitude,
                "longitude": request.longitude,
                "depth": request.depth
            },
            predictions=cached_result,
            cached=True,
            processing_time_ms=processing_time
        )
    
    # Make new prediction
    try:
        predictions = predictor.predict(
            request.latitude,
            request.longitude,
            request.depth
        )
        
        # Cache the result
        prediction_cache.set(
            request.latitude,
            request.longitude,
            request.depth,
            predictions
        )
        
        processing_time = (time.time() - start_time) * 1000
        metrics.record_prediction(cached=False, response_time=processing_time)
        
        logger.info(f"Prediction completed in {processing_time:.2f}ms")
        
        return PredictionResponse(
            success=True,
            timestamp=datetime.now(),
            input={
                "latitude": request.latitude,
                "longitude": request.longitude,
                "depth": request.depth
            },
            predictions=predictions,
            cached=False,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/api/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get application metrics and statistics"""
    metric_data = metrics.get_metrics()
    
    models_info = {}
    for target in predictor.results_summary:
        models_info[target] = {
            "model_type": predictor.results_summary[target]['best_model'],
            "r2_score": predictor.results_summary[target]['best_r2'],
            "features": predictor.results_summary[target].get('n_features', 0)
        }
    
    return MetricsResponse(
        total_predictions=metric_data['total_predictions'],
        cache_hit_rate=metric_data['cache_hit_rate'],
        average_response_time_ms=metric_data['average_response_time_ms'],
        models_info=models_info
    )

@app.post("/api/cache/clear", tags=["Admin"])
async def clear_cache():
    """Clear prediction cache (admin only)"""
    prediction_cache.clear()
    logger.info("Cache cleared via API")
    return {"success": True, "message": "Cache cleared"}

@app.get("/api/properties", tags=["Information"])
async def get_properties():
    """Get list of available properties"""
    return {
        "properties": predictor.target_columns,
        "feature_columns": predictor.feature_columns,
        "units": predictor.units
    }






@app.get("/api/model-info", tags=["Information"])
async def model_info():
    """Get detailed model information"""
    if not predictor.loaded:
        raise HTTPException(
            status_code=404,
            detail="Models not loaded"
        )
    
    info = {}
    for target in predictor.results_summary:
        info[target] = {
            "best_model": predictor.results_summary[target]['best_model'],
            "r2_score": predictor.results_summary[target]['best_r2'],
            "n_features": predictor.results_summary[target].get('n_features', 0),
            "unit": predictor.units.get(target, ''),
            "confidence": 'High' if predictor.results_summary[target]['best_r2'] > 0.8 
                         else 'Medium' if predictor.results_summary[target]['best_r2'] > 0.6 
                         else 'Low'
        }
    return info
@app.post("/api/generate-pdf", tags=["Reports"])
async def generate_pdf_endpoint(request: PredictionRequest):
    """Generate PDF report"""
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        predictions = predictor.predict(request.latitude, request.longitude, request.depth)
        
        pdf_data = {
            "input": {"latitude": request.latitude, "longitude": request.longitude, "depth": request.depth},
            "predictions": {
                key: {"value": pred.value, "r2_score": pred.r2_score, "confidence": pred.confidence, "unit": pred.unit}
                for key, pred in predictions.items() if pred.value is not None
            }
        }
        
        pdf_content = generate_pdf_report(pdf_data)
        filename = f"soil_report_{request.latitude}_{request.longitude}.pdf"
        
        return StreamingResponse(
            io.BytesIO(pdf_content),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=get_settings().DEBUG,
        log_level=get_settings().LOG_LEVEL.lower(),
        access_log=True
    )