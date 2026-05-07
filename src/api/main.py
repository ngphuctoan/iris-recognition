from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import SQLModel, Session, create_engine, select
import numpy as np
import cv2
import binascii
import os
import time

from src.api.models import User, IrisTemplate
from src.core.io import load_image
from src.core.segmentation import segment_iris, detect_eyelines
from src.core.normalization import normalize_iris
from src.core.encoding import encode_iris
from src.core.matching import match_codes

# 1. Setup Database
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

app = FastAPI(title="IrisGuard v3.0")

# Setup Templates and Static
script_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(script_dir, "templates"))

# 2. Dependency cho DB session
def get_session():
    with Session(engine) as session:
        yield session

import base64
import io

# 3. Helper functions
def get_base64_img(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def vis_segmentation(img, pupil, iris):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    px, py, pr = map(int, pupil)
    ix, iy, ir = map(int, iris)
    cv2.circle(vis, (ix, iy), ir, (133, 64, 0), 2) # Royal Blue (BGR)
    cv2.circle(vis, (px, py), pr, (102, 98, 0), 2) # Teal (BGR)
    return get_base64_img(vis)

def vis_iriscode(code):
    # Zoom in to make bits visible
    vis = cv2.resize(code * 255, (code.shape[1] * 4, code.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    return get_base64_img(vis)

def vis_hamming_match(code1, code2, mask1, mask2):
    """Green for match, Red for mismatch, Black for masked."""
    h, w = code1.shape
    combined_mask = cv2.bitwise_and(mask1, mask2)
    xor_res = cv2.bitwise_xor(code1, code2)
    
    # Create BGR image
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Matches (Green)
    vis[(combined_mask == 255) & (xor_res == 0)] = [0, 255, 0]
    # Mismatches (Red)
    vis[(combined_mask == 255) & (xor_res != 0)] = [0, 0, 255]
    
    # Resize for visibility
    vis_large = cv2.resize(vis, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
    return get_base64_img(vis_large)

def process_image(image_bytes: bytes):
    start_time = time.time()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Khong the giai ma anh")
        
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    p, i = segment_iris(img)
    mask_raw = detect_eyelines(img, p, i)
    norm, norm_mask = normalize_iris(img, p, i, mask=mask_raw)
    code, mask = encode_iris(norm, mask=norm_mask)
    
    compute_time = time.time() - start_time
    
    vis_seg = vis_segmentation(img, p, i)
    vis_code = vis_iriscode(code)
    
    return {
        "code": code,
        "mask": mask,
        "compute_time": compute_time,
        "vis_seg": vis_seg,
        "vis_code": vis_code
    }

# 4. Routes
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/", response_class=RedirectResponse)
async def root():
    return "/identify"

@app.get("/identify", response_class=HTMLResponse)
async def identify_get(request: Request):
    return templates.TemplateResponse(request, "identify.html", {"active_page": "identify"})

@app.post("/identify", response_class=HTMLResponse)
async def identify_post(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    try:
        contents = await file.read()
        p_res = process_image(contents)
        code_input, mask_input = p_res["code"], p_res["mask"]
        
        templates_db = session.exec(select(IrisTemplate)).all()
        
        best_match = None
        min_dist = 1.0
        best_code = None
        best_mask = None
        
        for temp in templates_db:
            t_code = np.frombuffer(binascii.unhexlify(temp.iris_code_hex), dtype=np.uint8).reshape(64, -1)
            t_mask = np.frombuffer(binascii.unhexlify(temp.mask_hex), dtype=np.uint8).reshape(64, -1)
            
            is_match, dist = match_codes(code_input, t_code, mask_input, t_mask)
            if dist < min_dist:
                min_dist = dist
                best_code = t_code
                best_mask = t_mask
                if is_match:
                    best_match = temp
        
        result = {
            "hamming_score": min_dist,
            "compute_time": p_res["compute_time"],
            "confidence": 1.0 - min_dist,
            "template_id": "TEMP_" + binascii.hexlify(os.urandom(4)).decode().upper(),
            "ocular_side": "UNKNOWN",
            "vis_seg": p_res["vis_seg"],
            "vis_code": p_res["vis_code"]
        }
        
        if best_match:
            result.update({
                "subject_name": best_match.user.name,
                "subject_id": best_match.user.identity_number,
                "verified": True,
                "ocular_side": best_match.eye_side.upper(),
                "vis_hamming": vis_hamming_match(code_input, best_code, mask_input, best_mask)
            })
        else:
            result.update({
                "subject_name": "Unknown Subject",
                "subject_id": "N/A",
                "verified": False,
                "vis_hamming": vis_hamming_match(code_input, best_code, mask_input, best_mask) if best_code is not None else None
            })
            
        return templates.TemplateResponse(request, "identify.html", {
            "active_page": "identify",
            "result": result
        })
    except Exception as e:
        return templates.TemplateResponse(request, "identify.html", {
            "active_page": "identify",
            "error": str(e)
        })

@app.get("/enroll", response_class=HTMLResponse)
async def enroll_get(request: Request):
    return templates.TemplateResponse(request, "enroll.html", {"active_page": "enroll"})

@app.post("/enroll/capture", response_class=HTMLResponse)
async def enroll_capture(
    request: Request,
    name: str = Form(...),
    subject_id: str = Form(...),
    eye: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    try:
        contents = await file.read()
        p_res = process_image(contents)
        
        code_hex = binascii.hexlify(p_res["code"].tobytes()).decode()
        mask_hex = binascii.hexlify(p_res["mask"].tobytes()).decode()
        
        # Get or Create User
        user = session.exec(select(User).where(User.identity_number == subject_id)).first()
        if not user:
            user = User(name=name, identity_number=subject_id)
            session.add(user)
            session.commit()
            session.refresh(user)
            
        template = IrisTemplate(
            user_id=user.id, eye_side=eye, iris_code_hex=code_hex, mask_hex=mask_hex
        )
        session.add(template)
        session.commit()
        
        return templates.TemplateResponse(request, "enroll.html", {
            "active_page": "enroll",
            "message": f"Biometric profile for {name} successfully committed to registry.",
            "vis_seg": p_res["vis_seg"],
            "vis_code": p_res["vis_code"]
        })
    except Exception as e:
        return templates.TemplateResponse(request, "enroll.html", {
            "active_page": "enroll",
            "error": str(e)
        })

@app.post("/enroll/sync", response_class=HTMLResponse)
async def enroll_sync(
    request: Request,
    sync_id: str = Form(...),
    sequence: str = Form(...),
    session: Session = Depends(get_session)
):
    try:
        # Load from dataset
        img = load_image(subject_id=sync_id, side="L", sequence=sequence) # Default to Left for sync
        
        # Process (Need a helper for img instead of bytes)
        # Reuse process_image logic but bypass decode
        start_time = time.time()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_p = clahe.apply(img)
        p, i = segment_iris(img_p)
        mask_raw = detect_eyelines(img_p, p, i)
        norm, norm_mask = normalize_iris(img_p, p, i, mask=mask_raw)
        code, mask = encode_iris(norm, mask=norm_mask)
        compute_time = time.time() - start_time
        
        code_hex = binascii.hexlify(code.tobytes()).decode()
        mask_hex = binascii.hexlify(mask.tobytes()).decode()
        
        # Auto-create User based on Sync ID
        user_id_str = f"CASIA-{sync_id}"
        user = session.exec(select(User).where(User.identity_number == user_id_str)).first()
        if not user:
            user = User(name=f"Subject_{sync_id}", identity_number=user_id_str)
            session.add(user)
            session.commit()
            session.refresh(user)
            
        template = IrisTemplate(
            user_id=user.id, eye_side="L", iris_code_hex=code_hex, mask_hex=mask_hex
        )
        session.add(template)
        session.commit()
        
        return templates.TemplateResponse(request, "enroll.html", {
            "active_page": "enroll",
            "message": f"Source sync complete for CASIA-{sync_id}. Template verified.",
            "vis_seg": vis_segmentation(img_p, p, i),
            "vis_code": vis_iriscode(code)
        })
    except Exception as e:
        return templates.TemplateResponse(request, "enroll.html", {
            "active_page": "enroll",
            "error": str(e)
        })

@app.get("/directory", response_class=HTMLResponse)
async def directory_get(request: Request, session: Session = Depends(get_session)):
    users = session.exec(select(User)).all()
    
    subjects_data = []
    for user in users:
        templates_user = user.templates
        subjects_data.append({
            "name": user.name,
            "subject_id": user.identity_number,
            "has_left": any(t.eye_side.lower() == "left" or t.eye_side.lower() == "l" for t in templates_user),
            "has_right": any(t.eye_side.lower() == "right" or t.eye_side.lower() == "r" for t in templates_user)
        })
        
    return templates.TemplateResponse(request, "directory.html", {
        "active_page": "directory",
        "subjects": subjects_data
    })
