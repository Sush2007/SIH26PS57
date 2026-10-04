"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  ArrowDown, ArrowUpRight, Check, ChevronRight, Compass,
  Crosshair, Download, Eye, EyeOff, FileImage, Focus, HelpCircle,
  Info, Layers3, MapPin, Radar, Radio, ScanLine,
  ShieldAlert, ShieldCheck, Upload, Waves, X,
} from "lucide-react";

type Target = {
  class_name: string;
  class_id: number;
  yolo_confidence: number;
  verdict: string;
  fused_score: number;
  bbox_obb: [number, number, number, number, number];
  evidence_breakdown: Record<string, number>;
  physical_dimensions: Record<string, number>;
  coordinates: { latitude?: number; longitude?: number };
  tactical_telemetry: {
    threat_classification?: string;
    acoustic_channel?: string;
    target_strike_heading_deg?: number;
    estimated_3d_relief_height_m?: number;
  };
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type EvidenceInfo = {
  key: string;
  plainTitle: string;
  technicalLabel: string;
  description: string;
  idealValue: string;
};

const evidenceDefinitions: EvidenceInfo[] = [
  {
    key: "calibrated_ai_probability",
    plainTitle: "AI Pattern Match",
    technicalLabel: "Calibrated AI Probability",
    description: "Statistical likelihood calculated by the neural network that an oriented underwater object exists here.",
    idealValue: "Higher is better (>65%)",
  },
  {
    key: "acoustic_shadow_strength",
    plainTitle: "Acoustic Shadow Verification",
    technicalLabel: "Acoustic Shadow Strength",
    description: "Real 3D seabed debris blocks sonar pings and casts an acoustic shadow behind it. Zero shadow indicates flat rocks or image artifacts.",
    idealValue: "Critical: Must be >15% for confirmed 3D objects",
  },
  {
    key: "image_quality_index",
    plainTitle: "Background Signal Clarity",
    technicalLabel: "Local SNR / Quality Index",
    description: "Evaluates seabed backscatter noise levels and acoustic speckle contrast in this sector.",
    idealValue: "Optimal range: 70% – 95%",
  },
  {
    key: "natural_feature_exclusion",
    plainTitle: "Man-Made vs Natural Filter",
    technicalLabel: "Natural Feature Exclusion",
    description: "Sobel gradient filter that penalizes natural seabed geological contours, ridges, and reefs to avoid false alarms.",
    idealValue: "100% = Clear artificial structure",
  },
  {
    key: "geometric_plausibility",
    plainTitle: "Target Dimension Conformity",
    technicalLabel: "Geometric Plausibility",
    description: "Validates whether measured dimensions strictly match known physical specifications for this debris category.",
    idealValue: "100% = Exactly matches military/maritime specs",
  },
];

const verdictExplanations: Record<string, { label: string; badgeClass: string; summary: string; action: string }> = {
  CONFIRMED_ANOMALY: {
    label: "CONFIRMED MARINE ANOMALY",
    badgeClass: "confirmed",
    summary: "Verified by both visual AI pattern recognition and physical acoustic shadow trigonometry. Extremely high probability of genuine seabed debris.",
    action: "Tactical target verified. Ready for recovery coordinate logging or ROV dive dispatch.",
  },
  PROBABLE_TARGET: {
    label: "PROBABLE CONTACT",
    badgeClass: "probable",
    summary: "Significant visual acoustic anomaly observed, but physical shadow verification or local SNR is borderline.",
    action: "Secondary sonar pass recommended from an orthogonal angle for shadow confirmation.",
  },
  REJECTED_NO_PHYSICAL_EVIDENCE: {
    label: "SUPPRESSED / NO SHADOW (FALSE POSITIVE)",
    badgeClass: "rejected",
    summary: "Flagged by the computer vision model, but the acoustic physics engine detected 0% acoustic shadow. Solid debris must block sonar waves; this is seabed noise or backscatter.",
    action: "Suppressed automatically to prevent false alarms. No action required.",
  },
  REJECTED_GEOMETRIC_VIOLATION: {
    label: "REJECTED (DIMENSION VIOLATION)",
    badgeClass: "rejected",
    summary: "The object's measured length/width severely violates realistic physical parameters for its detected category.",
    action: "Excluded from threat queue. Categorized as seabed distortion.",
  },
  REJECTED_AS_NATURAL_ARTIFACT: {
    label: "NATURAL SEABED FORMATION",
    badgeClass: "rejected",
    summary: "Texture and gradient analysis indicate natural geological formations (reef, rock outcrop, sand ripples) rather than synthetic marine debris.",
    action: "Filtered as natural background structure.",
  },
  REJECTED_AS_HARD_NEGATIVE: {
    label: "HARD NEGATIVE SUPPRESSED",
    badgeClass: "rejected",
    summary: "Identified as a non-hazardous commercial object (e.g. active crab pot) and automatically bypassed.",
    action: "Ignored by operational policy.",
  },
};

function percent(value: number) {
  return `${Math.max(0, Math.min(100, value * 100))}%`;
}

function verdictTone(verdict: string) {
  if (verdict === "CONFIRMED_ANOMALY") return "confirmed";
  if (verdict.startsWith("REJECTED_")) return "rejected";
  return "probable";
}

function displayNumber(value: number | undefined, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "N/A";
}

function getImageRenderMetrics(image: HTMLImageElement, containerWidth: number, containerHeight: number) {
  if (!image.naturalWidth || !image.naturalHeight || !containerWidth || !containerHeight) {
    return { renderWidth: containerWidth, renderHeight: containerHeight, offsetX: 0, offsetY: 0, scale: 1 };
  }
  const imgAspect = image.naturalWidth / image.naturalHeight;
  const containerAspect = containerWidth / containerHeight;

  let renderWidth = containerWidth;
  let renderHeight = containerHeight;
  let offsetX = 0;
  let offsetY = 0;

  if (containerAspect > imgAspect) {
    renderWidth = containerHeight * imgAspect;
    offsetX = (containerWidth - renderWidth) / 2;
  } else {
    renderHeight = containerWidth / imgAspect;
    offsetY = (containerHeight - renderHeight) / 2;
  }

  const scale = renderWidth / image.naturalWidth;
  return { renderWidth, renderHeight, offsetX, offsetY, scale };
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState("");
  const [targets, setTargets] = useState<Target[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [altitude, setAltitude] = useState("5.0");
  const [latitude, setLatitude] = useState("12.9716");
  const [longitude, setLongitude] = useState("77.5946");
  const [busy, setBusy] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [viewportVersion, setViewportVersion] = useState(0);
  const [filterVerifiedOnly, setFilterVerifiedOnly] = useState(false);
  const [activeTooltip, setActiveTooltip] = useState<string | null>(null);

  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const consoleRef = useRef<HTMLElement>(null);
  const imageUrlRef = useRef("");

  // Visible targets depending on filter
  const visibleTargets = filterVerifiedOnly
    ? targets.filter((t) => t.verdict === "CONFIRMED_ANOMALY" || t.verdict === "PROBABLE_TARGET")
    : targets;

  const selected = visibleTargets[selectedIndex] ?? visibleTargets[0] ?? targets[0];

  useEffect(() => {
    // 1. Silent pre-warm: Wake up the backend on page load so it's ready when the user uploads
    fetch(`${API_URL}/`, { method: "GET", mode: "cors" }).catch(() => {});

    // 2. Periodic keep-alive heartbeat while the tab is open (every 8 minutes)
    const heartbeat = setInterval(() => {
      fetch(`${API_URL}/`, { method: "GET", mode: "cors" }).catch(() => {});
    }, 8 * 60 * 1000);

    return () => {
      clearInterval(heartbeat);
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
    };
  }, []);

  // Sync canvas overlay with filtered targets
  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !image.complete || !image.naturalWidth) return;
    const bounds = image.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(bounds.width * dpr);
    canvas.height = Math.round(bounds.height * dpr);
    canvas.style.width = `${bounds.width}px`;
    canvas.style.height = `${bounds.height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, bounds.width, bounds.height);
    const { offsetX, offsetY, scale } = getImageRenderMetrics(image, bounds.width, bounds.height);

    visibleTargets.forEach((target, index) => {
      const [cx, cy, width, height, angle] = target.bbox_obb;
      const active = target === selected;
      const isRejected = target.verdict.startsWith("REJECTED_");
      const color = target.verdict === "CONFIRMED_ANOMALY" ? "#49e6b3" : isRejected ? "#dc7468" : "#f2b65e";

      const drawX = offsetX + cx * scale;
      const drawY = offsetY + cy * scale;
      const boxW = width * scale;
      const boxH = height * scale;

      ctx.save();
      ctx.translate(drawX, drawY);
      ctx.rotate(angle);
      ctx.strokeStyle = color;
      ctx.lineWidth = active ? 3.0 : 1.5;
      ctx.shadowColor = color;
      ctx.shadowBlur = active ? 16 : 5;
      if (isRejected) {
        ctx.globalAlpha = 0.65;
        ctx.setLineDash([4, 3]);
      }
      ctx.strokeRect(-boxW / 2, -boxH / 2, boxW, boxH);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, -boxH / 2 - 14);
      ctx.strokeStyle = "#38d6e8";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(drawX, drawY, active ? 4.5 : 2.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = "600 10px ui-monospace, monospace";
      const label = `T-${String(index + 1).padStart(2, "0")}`;
      let labelY = drawY + boxH / 2 + 6;
      if (labelY + 20 > bounds.height) {
        labelY = drawY - boxH / 2 - 20;
      }
      const labelX = Math.max(2, Math.min(bounds.width - 40, drawX - 18));
      ctx.fillStyle = "#081110";
      ctx.fillRect(labelX, labelY, 38, 17);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.strokeRect(labelX, labelY, 38, 17);
      ctx.fillStyle = color;
      ctx.fillText(label, labelX + 6, labelY + 12);
      ctx.restore();
    });
  }, [visibleTargets, selected, imageUrl, viewportVersion]);

  useEffect(() => {
    const stage = canvasRef.current?.parentElement;
    if (!stage) return;
    const observer = new ResizeObserver(() => setViewportVersion((version) => version + 1));
    observer.observe(stage);
    return () => observer.disconnect();
  }, [imageUrl]);

  function acceptFile(nextFile?: File) {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("image/")) {
      setError("Select a sonar image in JPG, PNG, or TIFF format.");
      return;
    }
    if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
    imageUrlRef.current = URL.createObjectURL(nextFile);
    setImageUrl(imageUrlRef.current);
    setFile(nextFile);
    setTargets([]);
    setHasRun(false);
    setError("");
    setSelectedIndex(0);
    window.setTimeout(() => consoleRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files[0]);
  }

  async function executeAnalysis() {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setHasRun(false);
    const body = new FormData();
    body.append("file", file);
    const query = new URLSearchParams({ altitude, towfish_lat: latitude, towfish_lon: longitude });
    try {
      const response = await fetch(`${API_URL}/api/v1/detect?${query}`, { method: "POST", body });
      if (!response.ok) {
        let errorMessage = `Analysis service returned ${response.status}. Check that the backend is running.`;
        try {
          const errorBody = await response.json();
          if (errorBody.detail) {
            errorMessage = errorBody.detail;
          }
        } catch {
          // fallback
        }
        throw new Error(errorMessage);
      }
      const result: Target[] = await response.json();
      setTargets(result);
      setSelectedIndex(0);
      setHasRun(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to reach the analysis service.");
    } finally {
      setBusy(false);
    }
  }

  function selectOnCanvas(event: React.MouseEvent<HTMLCanvasElement>) {
    const image = imageRef.current;
    if (!image || !image.naturalWidth || !image.naturalHeight) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    const { offsetX, offsetY, scale } = getImageRenderMetrics(image, rect.width, rect.height);
    const x = (clickX - offsetX) / scale;
    const y = (clickY - offsetY) / scale;

    const found = visibleTargets.findIndex((target) => {
      const [cx, cy, width, height, angle] = target.bbox_obb;
      const dx = x - cx;
      const dy = y - cy;
      const localX = dx * Math.cos(angle) + dy * Math.sin(angle);
      const localY = -dx * Math.sin(angle) + dy * Math.cos(angle);
      return Math.abs(localX) <= width / 2 + 10 && Math.abs(localY) <= height / 2 + 10;
    });
    if (found >= 0) setSelectedIndex(found);
  }

  function exportReportAsImage() {
    if (!targets.length) return;

    const reportCanvas = document.createElement("canvas");
    const width = 1200;
    const padding = 40;
    
    // Estimate height dynamically based on target count
    const headerHeight = 180;
    const summaryHeight = 160;
    const targetCardHeight = 220;
    const totalHeight = headerHeight + summaryHeight + (targets.length * targetCardHeight) + 120;
    
    reportCanvas.width = width;
    reportCanvas.height = totalHeight;
    const ctx = reportCanvas.getContext("2d");
    if (!ctx) return;

    // Background styling
    ctx.fillStyle = "#0a1217";
    ctx.fillRect(0, 0, width, totalHeight);

    // Subtle background tactical grid
    ctx.strokeStyle = "rgba(8, 126, 101, 0.08)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, totalHeight);
      ctx.stroke();
    }
    for (let y = 0; y < totalHeight; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Header banner
    ctx.fillStyle = "#087e65";
    ctx.fillRect(0, 0, width, 6);

    // Title & Branding
    ctx.font = "bold 24px monospace";
    ctx.fillStyle = "#49e6b3";
    ctx.fillText("AETHERSOUND AI // TACTICAL MISSION REPORT", padding, 55);

    ctx.font = "13px monospace";
    ctx.fillStyle = "#8fa6a0";
    ctx.fillText(`GENERATED: ${new Date().toUTCString()}  |  SURVEY FILE: ${file?.name ?? "UNNAMED"}`, padding, 85);
    ctx.fillText(`TOWFISH ALTITUDE: ${altitude}m  |  LAT: ${latitude}° N  |  LON: ${longitude}° E`, padding, 105);

    // Divider
    ctx.strokeStyle = "rgba(73, 230, 179, 0.25)";
    ctx.beginPath();
    ctx.moveTo(padding, 130);
    ctx.lineTo(width - padding, 130);
    ctx.stroke();

    // Summary Metrics Boxes
    const boxY = 150;
    const boxW = (width - padding * 2 - 45) / 4;
    const boxH = 90;

    const summaryCards = [
      { label: "TOTAL CANDIDATES", val: `${targets.length}`, col: "#ffffff", sub: "Neural Net Output" },
      { label: "VERIFIED HAZARDS", val: `${confirmedCount}`, col: "#49e6b3", sub: "Passed Shadow Physics" },
      { label: "PROBABLE CONTACTS", val: `${probableCount}`, col: "#f2b65e", sub: "Borderline Evidence" },
      { label: "FALSE ALARMS FILTERED", val: `${rejectedCount}`, col: "#dc7468", sub: "Zero Shadow / Artifacts" },
    ];

    summaryCards.forEach((c, i) => {
      const bx = padding + i * (boxW + 15);
      ctx.fillStyle = "#0f1c22";
      ctx.fillRect(bx, boxY, boxW, boxH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
      ctx.strokeRect(bx, boxY, boxW, boxH);

      // Top colored bar
      ctx.fillStyle = c.col;
      ctx.fillRect(bx, boxY, boxW, 3);

      ctx.font = "10px monospace";
      ctx.fillStyle = "#8fa6a0";
      ctx.fillText(c.label, bx + 15, boxY + 25);

      ctx.font = "bold 26px sans-serif";
      ctx.fillStyle = c.col;
      ctx.fillText(c.val, bx + 15, boxY + 60);

      ctx.font = "9px monospace";
      ctx.fillStyle = "#617a72";
      ctx.fillText(c.sub, bx + 15, boxY + 78);
    });

    // Target Dossier Section Header
    let currentY = 275;
    ctx.font = "bold 15px monospace";
    ctx.fillStyle = "#ffffff";
    ctx.fillText("INDIVIDUAL CONTACT INTELLIGENCE DOSSIERS", padding, currentY);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.beginPath();
    ctx.moveTo(padding, currentY + 12);
    ctx.lineTo(width - padding, currentY + 12);
    ctx.stroke();

    currentY += 30;

    // Render each target card
    targets.forEach((target, index) => {
      const cardY = currentY;
      const cardH = 200;
      const cardW = width - padding * 2;
      const isRejected = target.verdict.startsWith("REJECTED_");
      const themeColor = target.verdict === "CONFIRMED_ANOMALY" ? "#49e6b3" : isRejected ? "#dc7468" : "#f2b65e";

      // Card Background
      ctx.fillStyle = "#0e181e";
      ctx.fillRect(padding, cardY, cardW, cardH);
      ctx.strokeStyle = isRejected ? "rgba(220, 116, 104, 0.3)" : "rgba(73, 230, 179, 0.25)";
      ctx.strokeRect(padding, cardY, cardW, cardH);

      // Left Accent Border
      ctx.fillStyle = themeColor;
      ctx.fillRect(padding, cardY, 5, cardH);

      // Target ID & Name
      ctx.font = "bold 16px monospace";
      ctx.fillStyle = "#ffffff";
      ctx.fillText(`T-${String(index + 1).padStart(2, "0")}: ${target.class_name.replaceAll("_", " ").toUpperCase()}`, padding + 20, cardY + 30);

      // Verdict Pill
      ctx.fillStyle = isRejected ? "rgba(220, 116, 104, 0.2)" : "rgba(73, 230, 179, 0.2)";
      const verdictText = target.verdict.replaceAll("_", " ");
      const verdictWidth = ctx.measureText(verdictText).width + 20;
      ctx.fillRect(width - padding - verdictWidth - 15, cardY + 15, verdictWidth, 24);
      ctx.strokeStyle = themeColor;
      ctx.strokeRect(width - padding - verdictWidth - 15, cardY + 15, verdictWidth, 24);
      ctx.font = "bold 10px monospace";
      ctx.fillStyle = themeColor;
      ctx.fillText(verdictText, width - padding - verdictWidth - 5, cardY + 31);

      // Row 1: Tactical Classification & Telemetry
      ctx.font = "11px monospace";
      ctx.fillStyle = "#8fa6a0";
      ctx.fillText(`THREAT: ${target.tactical_telemetry?.threat_classification ?? "UNKNOWN"}`, padding + 20, cardY + 60);
      ctx.fillText(`ACOUSTIC BEAM: ${target.tactical_telemetry?.acoustic_channel ?? "STARBOARD"}`, padding + 340, cardY + 60);
      ctx.fillText(`FUSED CONFIDENCE: ${Math.round(target.fused_score * 100)}% (RAW YOLO: ${(target.yolo_confidence * 100).toFixed(1)}%)`, padding + 620, cardY + 60);

      // Row 2: Physical Dimensions & 3D Protrusion
      ctx.fillStyle = "#d0deda";
      ctx.fillText(`LENGTH: ${displayNumber(target.physical_dimensions?.length_meters)}m`, padding + 20, cardY + 90);
      ctx.fillText(`WIDTH: ${displayNumber(target.physical_dimensions?.width_meters)}m`, padding + 170, cardY + 90);
      ctx.fillText(`GROUND RANGE (Rg): ${displayNumber(target.physical_dimensions?.ground_range_meters)}m`, padding + 340, cardY + 90);
      ctx.fillText(`PROTRUSION HEIGHT (H): ${displayNumber(target.tactical_telemetry?.estimated_3d_relief_height_m)}m`, padding + 620, cardY + 90);

      // Row 3: Coordinates
      ctx.fillStyle = "#8ce0c6";
      ctx.fillText(`GEODETIC POSITION: LAT ${displayNumber(target.coordinates?.latitude, 6)}° N  |  LON ${displayNumber(target.coordinates?.longitude, 6)}° E`, padding + 20, cardY + 120);

      // Row 4: Evidence Scores
      ctx.font = "10px monospace";
      ctx.fillStyle = "#718780";
      ctx.fillText("6-PILLAR PHYSICS BREAKDOWN:", padding + 20, cardY + 155);

      const ev = target.evidence_breakdown || {};
      const evList = [
        `AI: ${Math.round((ev.calibrated_ai_probability ?? 0) * 100)}%`,
        `Shadow: ${Math.round((ev.acoustic_shadow_strength ?? 0) * 100)}%`,
        `SNR: ${Math.round((ev.image_quality_index ?? 0) * 100)}%`,
        `Non-Rock: ${Math.round((ev.natural_feature_exclusion ?? 0) * 100)}%`,
        `Geometry: ${Math.round((ev.geometric_plausibility ?? 0) * 100)}%`,
      ];
      ctx.fillStyle = "#ffffff";
      ctx.fillText(evList.join("   |   "), padding + 20, cardY + 175);

      currentY += cardH + 18;
    });

    // Footer
    ctx.font = "10px monospace";
    ctx.fillStyle = "#4a635c";
    ctx.fillText("AETHERSOUND AI // SIH 2026 PS 26057 // AUTONOMOUS ACOUSTIC INTELLIGENCE DOSSIER", padding, totalHeight - 35);
    ctx.fillText("PAGE 1 OF 1 // CLASSIFIED OCEAN TELEMETRY", width - padding - 310, totalHeight - 35);

    // Download as PNG image
    reportCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `sonar_mission_report_${Date.now()}.png`;
      link.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  }

  function jumpToConsole() {
    consoleRef.current?.scrollIntoView({ behavior: "smooth" });
    window.setTimeout(() => fileRef.current?.click(), 250);
  }

  // Summary counts
  const confirmedCount = targets.filter((t) => t.verdict === "CONFIRMED_ANOMALY").length;
  const probableCount = targets.filter((t) => t.verdict === "PROBABLE_TARGET").length;
  const rejectedCount = targets.filter((t) => t.verdict.startsWith("REJECTED_")).length;

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Aether Sound AI home">
          <span className="brand-mark"><Waves size={18} strokeWidth={1.8} /></span>
          <span>AETHER<span className="brand-light">SOUND</span> AI</span>
        </a>
        <div className="topbar-right">
          <div className="system-status">
            <span className="live-dot" />
            <span>SONAR ENGINE ONLINE</span>
          </div>
          <button className="nav-upload" onClick={jumpToConsole}>INGEST SONAR LOG <ArrowUpRight size={14} /></button>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-grid" />
        
        {/* Animated Pure-CSS Tactical Sonar Radar Sweep */}
        <div className="sonar-visual" aria-hidden="true">
          <div className="sonar-ring ring-one" />
          <div className="sonar-ring ring-two" />
          <div className="sonar-ring ring-three" />
          <div className="sonar-sweep" />
          <div className="sonar-origin" />
          <div className="sonar-contact contact-one" />
          <div className="sonar-contact contact-two" />
          <span className="range-label range-a">RANGE 50M</span>
          <span className="range-label range-b">RANGE 100M</span>
          <div className="ghost-vessel"><span /><i></i><b></b></div>
        </div>

        <div className="hero-content">
          <div className="eyebrow"><span className="eyebrow-line" /> AUTONOMOUS MINE &amp; DEBRIS COUNTERMEASURES</div>
          <h1>ILLUMINATING THE<br /><span>ABYSSAL SILENCE.</span></h1>
          <p className="hero-copy">Autonomous sonar intelligence detecting ghost nets, benthic debris, and submerged hazards across Indian EEZ waters with millimetric precision and acoustic shadow physics validation.</p>
          <div className="hero-actions">
            <button className="primary-button" onClick={jumpToConsole}>UPLOAD YOUR SONAR LOG <ArrowUpRight size={16} /></button>
            <a className="secondary-button" href="#console"><span className="play-icon"><ArrowDown size={13} /></span> EXPLORE THE CONSOLE</a>
          </div>
        </div>
        <div className="hero-index"><span>01</span><i /> <span>02</span></div>
        <a href="#console" className="scroll-cue"><span>SCROLL TO ANALYZE</span><ArrowDown size={13} /></a>
      </section>

      <section className="console-section" ref={consoleRef} id="console">
        <div className="console-heading">
          <div>
            <span className="section-kicker">FIELD CONSOLE / TACTICAL ACOUSTIC SWEEP</span>
            <h2>Acoustic Telemetry &amp; Physics Validation</h2>
          </div>
        </div>

        <div className="flight-bar">
          <div className="flight-title"><Compass size={16} /><span>SURVEY PARAMETERS</span></div>
          <label className="field-group" title="Vertical altitude of the sonar towfish above the seabed">
            <span>ALTITUDE <small>M</small> <Info size={11} className="inline-info" /></span>
            <input aria-label="Altitude in meters" type="number" step="0.1" min="0" value={altitude} onChange={(event) => setAltitude(event.target.value)} />
          </label>
          <label className="field-group" title="Towfish geodetic latitude at the moment of scan acquisition">
            <span>TOWFISH LAT <Info size={11} className="inline-info" /></span>
            <input aria-label="Towfish latitude" type="number" step="0.0001" value={latitude} onChange={(event) => setLatitude(event.target.value)} />
          </label>
          <label className="field-group" title="Towfish geodetic longitude at the moment of scan acquisition">
            <span>TOWFISH LON <Info size={11} className="inline-info" /></span>
            <input aria-label="Towfish longitude" type="number" step="0.0001" value={longitude} onChange={(event) => setLongitude(event.target.value)} />
          </label>
          <button className="execute-button" disabled={!file || busy} onClick={executeAnalysis}>
            {busy ? <><span className="button-spinner" /> RUNNING 6-PILLAR ENGINE...</> : <><ScanLine size={15} /> EXECUTE ANALYSIS</>}
          </button>
        </div>

        <input ref={fileRef} className="visually-hidden" type="file" accept="image/*" onChange={onFileChange} />
        {error && <div className="error-banner" role="alert"><ShieldAlert size={16} /> <div><strong>Scan Rejected:</strong> {error}</div></div>}

        {!file ? (
          <div className={`ingest-zone ${dragging ? "is-dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
            <div className="ingest-icon"><FileImage size={22} /></div>
            <div>
              <h3>Awaiting Side-Scan Sonar Imagery</h3>
              <p>Drop a raw sonar scan file here, or browse your directory</p>
            </div>
            <button className="browse-button" onClick={() => fileRef.current?.click()}><Upload size={15} /> CHOOSE SONAR SCAN</button>
            <span className="file-hint">JPG · PNG · TIFF · Real Sonar Logs Only (AI Protected Against Non-Sonar Imagery)</span>
          </div>
        ) : (
          <>
            {/* Mission Overview Ribbon */}
            {hasRun && (
              <div className="mission-summary-ribbon">
                <div className="summary-card">
                  <span className="summary-title">TOTAL CANDIDATES</span>
                  <strong className="summary-number">{targets.length}</strong>
                  <span className="summary-sub">Detected by Neural Net</span>
                </div>
                <div className="summary-card confirmed-border">
                  <span className="summary-title">VERIFIED HAZARDS</span>
                  <strong className="summary-number text-green">{confirmedCount}</strong>
                  <span className="summary-sub">Passed Shadow Physics</span>
                </div>
                <div className="summary-card probable-border">
                  <span className="summary-title">PROBABLE CONTACTS</span>
                  <strong className="summary-number text-amber">{probableCount}</strong>
                  <span className="summary-sub">Borderline Shadow Evidence</span>
                </div>
                <div className="summary-card rejected-border">
                  <span className="summary-title">FALSE POSITIVES FILTERED</span>
                  <strong className="summary-number text-red">{rejectedCount}</strong>
                  <span className="summary-sub">Zero Shadow / Speckle Noise</span>
                </div>
                <div className="summary-card actions-card">
                  <button className="export-report-btn" onClick={exportReportAsImage} title="Export full graphical tactical report as PNG image">
                    <Download size={14} /> EXPORT MISSION REPORT (PNG)
                  </button>
                  <button
                    className={`toggle-filter-btn ${filterVerifiedOnly ? "active" : ""}`}
                    onClick={() => {
                      setFilterVerifiedOnly(!filterVerifiedOnly);
                      setSelectedIndex(0);
                    }}
                  >
                    {filterVerifiedOnly ? <><Eye size={13} /> SHOWING VERIFIED ONLY</> : <><EyeOff size={13} /> SHOW ALL (INCL. REJECTED)</>}
                  </button>
                </div>
              </div>
            )}

            <div className="workspace-grid">
              <div className="viewport-column">
                <div className="viewport-toolbar">
                  <div><span className="live-dot" /> SONAR VIEWPORT <span className="toolbar-divider">/</span> {file.name}</div>
                  <button className="icon-button" aria-label="Remove image" title="Remove image" onClick={() => { if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current); imageUrlRef.current = ""; setImageUrl(""); setFile(null); setTargets([]); setHasRun(false); }}>
                    <X size={15} />
                  </button>
                </div>

                <div className="image-stage" onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
                  <img ref={imageRef} src={imageUrl} alt="Uploaded sonar scan" onLoad={() => setViewportVersion((version) => version + 1)} />
                  <canvas ref={canvasRef} className="obb-canvas" aria-label="Detection overlay. Click a target box to inspect." onClick={selectOnCanvas} />
                  {!hasRun && <div className="awaiting-overlay"><Focus size={19} /><span>IMAGE LOADED · READY FOR ANALYSIS</span></div>}
                  {busy && <div className="scan-overlay"><div className="scan-line" /><span><Radar size={16} /> RUNNING 6-PILLAR DUAL-STREAM FUSION...</span></div>}
                  <div className="viewport-corner corner-tl" /><div className="viewport-corner corner-tr" /><div className="viewport-corner corner-bl" /><div className="viewport-corner corner-br" />
                </div>

                <div className="viewport-footer">
                  <span>
                    <Crosshair size={13} />
                    {visibleTargets.length
                      ? `${visibleTargets.length} CONTACT${visibleTargets.length === 1 ? "" : "S"} VISIBLE (${filterVerifiedOnly ? "VERIFIED FILTER ACTIVE" : "ALL CANDIDATES"})`
                      : hasRun
                        ? "NO DETECTIONS IN THIS VIEW"
                        : "AWAITING SWEEP"}
                  </span>
                  <span>CLICK ANY TARGET BOX OR CARD TO INSPECT</span>
                </div>

                {hasRun && visibleTargets.length > 0 && (
                  <div className="target-list">
                    <div className="target-list-header">
                      <span className="target-list-label">DETECTED CONTACTS</span>
                      <span className="target-list-hint">Click to focus details</span>
                    </div>
                    {visibleTargets.map((target, index) => {
                      const isRejected = target.verdict.startsWith("REJECTED_");
                      return (
                        <button
                          className={`target-row ${selected === target ? "active" : ""} ${isRejected ? "target-row-rejected" : ""}`}
                          key={`${target.class_id}-${index}`}
                          onClick={() => setSelectedIndex(index)}
                        >
                          <span className={`target-dot ${verdictTone(target.verdict)}`} />
                          <div className="target-row-info">
                            <span className="target-row-name">
                              T-{String(index + 1).padStart(2, "0")} / {target.class_name.replaceAll("_", " ").toUpperCase()}
                            </span>
                            <span className="target-row-subtitle">
                              {target.verdict === "CONFIRMED_ANOMALY" ? "Confirmed Hazard" : target.verdict === "PROBABLE_TARGET" ? "Probable Anomaly" : "Suppressed (No Shadow)"}
                            </span>
                          </div>
                          <span className="target-row-score">{Math.round(target.fused_score * 100)}%</span>
                          <ChevronRight size={14} />
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              <aside className="telemetry-panel">
                <div className="panel-heading">
                  <div>
                    <span className="section-kicker">ACOUSTIC INTELLIGENCE DOSSIER</span>
                    <h3>Target Telemetry &amp; Physics Verdict</h3>
                  </div>
                  <Layers3 size={17} />
                </div>

                {!hasRun ? (
                  <div className="panel-placeholder">
                    <Radar size={28} />
                    <p>Execute analysis to unpack the 6-pillar physics validation, 3D relief height, and GPS seabed coordinates.</p>
                  </div>
                ) : visibleTargets.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-state-icon"><ShieldCheck size={24} /></div>
                    <span className="empty-state-tag"><Check size={12} /> SWEEP COMPLETE</span>
                    <h3>No Hazards Detected</h3>
                    <p>The acoustic physics engine found no verified marine debris matching the active filter.</p>
                  </div>
                ) : selected ? (
                  <div className="telemetry-content">
                    {/* 01 Tactical Verdict Card */}
                    <article className="telemetry-card verification-card">
                      <div className="card-title">
                        <span>01</span> VERIFICATION &amp; TACTICAL VERDICT
                      </div>
                      <div className="target-name-row">
                        <div>
                          <h4>{selected.class_name.replaceAll("_", " ")}</h4>
                          <span className="class-id">CATEGORY ID #{selected.class_id}</span>
                        </div>
                        <span className={`verdict-badge ${verdictTone(selected.verdict)}`}>
                          {verdictExplanations[selected.verdict]?.label ?? selected.verdict.replaceAll("_", " ")}
                        </span>
                      </div>

                      {/* Plain-English Interpretive Banner */}
                      <div className={`verdict-explanation-box ${verdictTone(selected.verdict)}`}>
                        <div className="explanation-summary">
                          <strong>Operational Impact:</strong> {verdictExplanations[selected.verdict]?.summary}
                        </div>
                        <div className="explanation-action">
                          <strong>Recommended Protocol:</strong> {verdictExplanations[selected.verdict]?.action}
                        </div>
                      </div>

                      <div className="tag-row">
                        <span className="tactical-tag">
                          <ShieldCheck size={12} /> {selected.tactical_telemetry?.threat_classification ?? "HAZARD"}
                        </span>
                        <span className="channel-tag">
                          <Radio size={12} /> {selected.tactical_telemetry?.acoustic_channel ?? "CHANNEL"} ACOUSTIC BEAM
                        </span>
                      </div>

                      <div className="score-block">
                        <div className="score-head">
                          <div>
                            <span>FUSED ACOUSTIC CONFIDENCE</span>
                            <small className="score-caption">Combined AI + Physics Multi-Evidence</small>
                          </div>
                          <strong>{Math.round(selected.fused_score * 100)}<small>%</small></strong>
                        </div>
                        <div className="progress-track score-track">
                          <span style={{ width: percent(selected.fused_score) }} />
                        </div>
                      </div>

                      <div className="baseline-row">
                        <span>RAW YOLO DETECTION CONFIDENCE (AI Only)</span>
                        <strong>{(selected.yolo_confidence * 100).toFixed(1)}%</strong>
                      </div>
                    </article>

                    {/* 02 Physics Evidence Breakdown */}
                    <article className="telemetry-card">
                      <div className="card-title">
                        <span>02</span> 6-PILLAR ACOUSTIC EVIDENCE BREAKDOWN
                      </div>
                      <p className="card-desc">
                        Hover the information icons to see how our physics engine verifies this target beyond standard computer vision:
                      </p>
                      <div className="evidence-list">
                        {evidenceDefinitions.map((item) => {
                          const score = selected.evidence_breakdown[item.key];
                          const isShadow = item.key === "acoustic_shadow_strength";
                          const isZeroShadow = isShadow && (score === 0 || score == null);
                          return (
                            <div className={`evidence-item ${isZeroShadow ? "shadow-warning-item" : ""}`} key={item.key}>
                              <div className="evidence-label">
                                <div className="evidence-title-group">
                                  <span>{item.plainTitle}</span>
                                  <button
                                    type="button"
                                    className="info-bubble-btn"
                                    onClick={() => setActiveTooltip(activeTooltip === item.key ? null : item.key)}
                                    title={item.description}
                                  >
                                    <HelpCircle size={12} />
                                  </button>
                                </div>
                                <strong>{score == null ? "N/A" : <>{(score * 100).toFixed(0)}<small>%</small></>}</strong>
                              </div>

                              {activeTooltip === item.key && (
                                <div className="interactive-tooltip">
                                  <div className="tooltip-tech"><strong>Technical Name:</strong> {item.technicalLabel}</div>
                                  <div className="tooltip-body">{item.description}</div>
                                  <div className="tooltip-ideal"><strong>Benchmark:</strong> {item.idealValue}</div>
                                </div>
                              )}

                              <div className="progress-track">
                                <span style={{ width: score == null ? "0%" : percent(score) }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      <div className="uncertainty-row">
                        <div>
                          <span>GEOSPATIAL POSITIONING UNCERTAINTY</span>
                          <small className="field-subtext">Estimated GPS margin of error on seafloor</small>
                        </div>
                        <strong>±{displayNumber(selected.evidence_breakdown.geospatial_accuracy_radius_m)} <small>m</small></strong>
                      </div>
                    </article>

                    {/* 03 Physical Dimensions & 3D Relief */}
                    <article className="telemetry-card">
                      <div className="card-title">
                        <span>03</span> PHYSICAL DIMENSIONS &amp; 3D RELIEF
                      </div>
                      <div className="metric-grid">
                        <div className="metric-cell" title="Longest physical dimension on seabed">
                          <span>OBJECT LENGTH</span>
                          <strong>{displayNumber(selected.physical_dimensions.length_meters)}<small> m</small></strong>
                        </div>
                        <div className="metric-cell" title="Shortest physical dimension on seabed">
                          <span>OBJECT WIDTH</span>
                          <strong>{displayNumber(selected.physical_dimensions.width_meters)}<small> m</small></strong>
                        </div>
                        <div className="metric-cell" title="True horizontal seabed distance from the central nadir line">
                          <span>GROUND RANGE (<i>R<sub>g</sub></i>)</span>
                          <strong>{displayNumber(selected.physical_dimensions.ground_range_meters)}<small> m</small></strong>
                        </div>
                        <div className="metric-cell highlight-cell" title="Calculated protrusion height above the seafloor using acoustic shadow trigonometry">
                          <span>PROTRUSION HEIGHT (<i>H</i>)</span>
                          <strong>{displayNumber(selected.tactical_telemetry.estimated_3d_relief_height_m)}<small> m</small></strong>
                        </div>
                      </div>
                      <p className="relief-note">
                        💡 <strong>Protrusion Height ($H$):</strong> Calculated via shadow trigonometry $H = \frac{'{L_s \\cdot A}{L_s + R}'}$. Real seabed debris sticks up and casts shadows; flat debris sits level.
                      </p>
                      <div className="heading-row">
                        <span><Compass size={13} /> STRIKE ORIENTATION HEADING</span>
                        <strong>{displayNumber(selected.tactical_telemetry.target_strike_heading_deg, 1)}°</strong>
                      </div>
                    </article>

                    {/* 04 Geodetic GPS Coordinates */}
                    <article className="telemetry-card geo-card">
                      <div className="card-title">
                        <span>04</span> GEODETIC SEABED POSITION (GPS)
                      </div>
                      <p className="card-desc">
                        Calculated from towfish navigation telemetry + slant-to-ground range trigonometry:
                      </p>
                      <div className="coordinate-row">
                        <MapPin size={14} />
                        <span>TARGET LATITUDE</span>
                        <strong>{displayNumber(selected.coordinates.latitude, 6)}{selected.coordinates.latitude == null ? "" : "° N"}</strong>
                      </div>
                      <div className="coordinate-row">
                        <MapPin size={14} />
                        <span>TARGET LONGITUDE</span>
                        <strong>{displayNumber(selected.coordinates.longitude, 6)}{selected.coordinates.longitude == null ? "" : "° E"}</strong>
                      </div>
                      {typeof selected.coordinates.latitude === "number" && typeof selected.coordinates.longitude === "number" && (
                        <a
                          className="map-link"
                          href={`https://www.openstreetmap.org/?mlat=${selected.coordinates.latitude}&mlon=${selected.coordinates.longitude}#map=18/${selected.coordinates.latitude}/${selected.coordinates.longitude}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          OPEN TARGET LOCATION IN OPENSTREETMAP <ArrowUpRight size={14} />
                        </a>
                      )}
                    </article>
                  </div>
                ) : null}
              </aside>
            </div>
          </>
        )}
      </section>

      <footer className="footer">
        <a className="brand footer-brand" href="#top">
          <span className="brand-mark"><Waves size={16} /></span>
          <span>AETHER<span className="brand-light">SOUND</span> AI</span>
        </a>
        <span>ACOUSTIC INTELLIGENCE FOR A CLEARER OCEAN</span>
        <span>SIH 2026 <i>/</i> PS 26057</span>
      </footer>
    </main>
  );
}
