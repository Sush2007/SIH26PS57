from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
from pathlib import Path

from utils.preprocessing import preprocess_sonar_pipeline
from utils.decision_engine import MultiEvidenceEngine

app = FastAPI(title="Sonar Debris Multi-Evidence Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    processed_image = preprocess_sonar_pipeline(image_bytes)
    print(f"🖼️ Preprocessing finished: Image shape {processed_image.shape}")
    
    # Run YOLO with conf=0.25 to see low-confidence candidates
    CONF_THRESHOLD = 0.25
    results = model.predict(processed_image, imgsz=640, conf=CONF_THRESHOLD)
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
        
        # 🧠 Run all 6 pillars of evidence through the decision engine
        decision = engine.evaluate(
            img=processed_image,
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
        
        # 🚫 Check rejection filter
        if verdict == "REJECTED_AS_NATURAL_ARTIFACT":
            print(f"  ⛔ DROPPED: Filtered out as natural seabed/artifact.")
            continue
            
        print(f"  ✅ ACCEPTED: Added to final payload.")
        final_output.append(
            DetailedAssessment(
                class_name=class_name,
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
        
    print(f"\n🏁 [PIPELINE COMPLETE] Returning {len(final_output)} verified target(s)")
    print("=" * 55 + "\n")
    return final_output