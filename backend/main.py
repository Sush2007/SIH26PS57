import os
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path

try:
    from backend.utils.preprocessing import preprocess_sonar_pipeline
    from backend.utils.decision_engine import MultiEvidenceEngine
    from backend.utils.sonar_validator import validate_sonar_image
except ModuleNotFoundError:
    from utils.preprocessing import preprocess_sonar_pipeline
    from utils.decision_engine import MultiEvidenceEngine
    from utils.sonar_validator import validate_sonar_image

app = FastAPI(title="Sonar Debris Multi-Evidence Engine")

# Robust CORS configuration:
# 1. Matches ALLOWED_ORIGINS env var if provided (e.g. from Render dashboard)
# 2. Includes allow_origin_regex matching all Vercel deployments (*.vercel.app) and localhost
# 3. Falls back to ["*"] with allow_credentials=False for absolute open access if requested
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env and allowed_origins_env.strip() != "*":
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
    allow_creds = True
else:
    # If ALLOWED_ORIGINS is not set or is "*", open to all origins
    allowed_origins = ["*"]
    allow_creds = False  # Wildcard '*' requires allow_credentials=False per CORS specification

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+",
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
def health_check():
    """Quick ping endpoint for Render health checks and uptime monitors."""
    return {"status": "online", "service": "AetherSound AI Backend"}

engine = MultiEvidenceEngine(meters_per_pixel=0.05)

BASE_DIR = Path(__file__).resolve().parent
model = YOLO(str(BASE_DIR / "models" / "sonar_model.onnx"), task="obb")

CLASS_NAMES = {
    0: "crab_pot",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder"
}

class DetailedAssessment(BaseModel):
    class_name: str
    class_id: int
    yolo_confidence: float
    verdict: str
    fused_score: float
    bbox_obb: List[float]
    evidence_breakdown: Dict[str, float]
    physical_dimensions: Dict[str, float]
    coordinates: Dict[str, float]
    tactical_telemetry: Dict[str, Any]

@app.post("/api/v1/detect", response_model=List[DetailedAssessment])
async def detect(
    file: UploadFile = File(...), 
    altitude: float = 5.0, 
    towfish_lat: float = 12.9716, 
    towfish_lon: float = 77.5946
):
    print("\n" + "=" * 55)
    print(f"📥 [PIPELINE START] Received file: {file.filename}")
    print(f"📍 Towfish params: altitude={altitude}m, lat={towfish_lat}, lon={towfish_lon}")
    
    image_bytes = await file.read()
    
    # 🛡️ INPUT VALIDATION: Reject non-sonar images before they reach the model
    np_arr = np.frombuffer(image_bytes, np.uint8)
    raw_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if raw_gray is None:
        raise HTTPException(status_code=422, detail="Could not decode image. Please upload a valid image file.")
    
    validation = validate_sonar_image(raw_gray)
    print(f"🛡️ [SONAR VALIDATOR] Score: {validation['confidence']:.2f} | Valid: {validation['is_valid']}")
    if not validation["is_valid"]:
        print(f"🚫 [REJECTED] {validation['reason']}")
        print(f"   Checks: {validation['checks']}")
        print("=" * 55 + "\n")
        raise HTTPException(
            status_code=422,
            detail=f"This does not appear to be a sonar image. {validation['reason']}. "
                   f"(Sonar confidence: {validation['confidence']:.0%}). "
                   f"Please upload a valid side-scan sonar scan."
        )
    
    # 🔀 Unpack Decoupled Dual-Stream (Visual for AI, Radiometric for Physics)
    visual_image, radiometric_image = preprocess_sonar_pipeline(image_bytes)
    print(f"🖼️ Dual-Stream Preprocessing finished: Shape {visual_image.shape}")
    
    # Run YOLO on Stream A (Visual CLAHE Stream)
    CONF_THRESHOLD = 0.25
    results = model.predict(visual_image, imgsz=640, conf=CONF_THRESHOLD)
    result = results[0]  # type: ignore
    
    final_output = []
    
    # 🔍 Check raw detections
    if result.obb is None or len(result.obb) == 0:  # type: ignore
        print(f"❌ [YOLO RESULTS] 0 OBB detections found at conf >= {CONF_THRESHOLD}")
        print("=" * 55 + "\n")
        return []

    num_detections = len(result.obb)  # type: ignore
    print(f"🎯 [YOLO RESULTS] Found {num_detections} raw OBB candidate(s):")

    for idx, obb in enumerate(result.obb):  # type: ignore
        xywhr = obb.xywhr[0].cpu().numpy().tolist()
        raw_conf = float(obb.conf[0].cpu().numpy())
        cls_id = int(obb.cls[0].cpu().numpy())
        class_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
        
        print(f"\n  --- Candidate #{idx + 1} ---")
        print(f"  🏷️ Class: {class_name} (ID: {cls_id})")
        print(f"  📊 Raw Confidence: {raw_conf:.3f}")
        print(f"  📐 OBB [x, y, w, h, r]: {[round(x, 2) for x in xywhr]}")
        
        # 🧠 Run 6 pillars of evidence on Stream B (Pure Radiometric Stream)
        decision = engine.evaluate(
            img=radiometric_image,
            obb_xywhr=xywhr,
            class_name=class_name,
            altitude_m=altitude,
            towfish_lat=towfish_lat,
            towfish_lon=towfish_lon,
            raw_conf=raw_conf
        )
        
        verdict = decision.get("verdict", "UNKNOWN")
        fused = decision.get("fused_confidence", 0.0)
        print(f"  ⚖️ Decision Engine Verdict: {verdict}")
        print(f"  ⭐ Fused Score: {fused:.3f}")
        print(f"  🔬 Evidence Breakdown: {decision.get('evidence_breakdown', {})}")
        
        print(f"  📋 RETAINED: Candidate included in final report.")
        final_output.append(
            DetailedAssessment(
                class_name=class_name,
                class_id=cls_id,
                yolo_confidence=round(raw_conf, 3),
                verdict=verdict,
                fused_score=fused,
                bbox_obb=xywhr,
                evidence_breakdown=decision["evidence_breakdown"],
                physical_dimensions=decision["physical_dimensions"],
                coordinates=decision["corrected_coordinates"],
                tactical_telemetry=decision["tactical_telemetry"]
            )
        )
        
    print(f"\n🏁 [PIPELINE COMPLETE] Returning {len(final_output)} model candidate(s)")
    print("=" * 55 + "\n")
    return final_output