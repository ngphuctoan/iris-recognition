from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sqlmodel import SQLModel, Session, create_engine, select
import numpy as np
import cv2
import binascii
import os

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

app = FastAPI(title="Iris Recognition System")

# Mount static files
script_dir = os.path.dirname(__file__)
static_dir = os.path.join(script_dir, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

# 2. Dependency cho DB session
def get_session():
    with Session(engine) as session:
        yield session

# 3. Helper functions
def process_image(image_bytes: bytes):
    # Chuyen bytes thanh numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Khong the giai ma anh")
        
    # Ap dung CLAHE nhu trong io.py
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Pipeline
    p, i = segment_iris(img)
    mask_raw = detect_eyelines(img, p, i)
    norm, norm_mask = normalize_iris(img, p, i, mask=mask_raw)
    code, mask = encode_iris(norm, mask=norm_mask)
    
    return code, mask

# 4. API Endpoints
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/enroll")
async def enroll(
    name: str = Form(...),
    identity_number: str = Form(...),
    eye_side: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    try:
        contents = await file.read()
        code, mask = process_image(contents)
        
        # Chuyen sang hex de luu tru
        code_hex = binascii.hexlify(code.tobytes()).decode()
        mask_hex = binascii.hexlify(mask.tobytes()).decode()
        
        # Kiem tra xem user da ton tai chua
        user = session.exec(select(User).where(User.identity_number == identity_number)).first()
        if not user:
            user = User(name=name, identity_number=identity_number)
            session.add(user)
            session.commit()
            session.refresh(user)
            
        template = IrisTemplate(
            user_id=user.id,
            eye_side=eye_side,
            iris_code_hex=code_hex,
            mask_hex=mask_hex
        )
        session.add(template)
        session.commit()
        
        return {"status": "success", "message": f"Da dang ky {name} thanh cong"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    try:
        contents = await file.read()
        code_input, mask_input = process_image(contents)
        
        # Lay tat ca template trong DB
        templates = session.exec(select(IrisTemplate)).all()
        
        best_match = None
        min_dist = 1.0
        
        for temp in templates:
            # Decode tu hex
            t_code = np.frombuffer(binascii.unhexlify(temp.iris_code_hex), dtype=np.uint8).reshape(64, -1)
            t_mask = np.frombuffer(binascii.unhexlify(temp.mask_hex), dtype=np.uint8).reshape(64, -1)
            
            is_match, dist = match_codes(code_input, t_code, mask_input, t_mask)
            if dist < min_dist:
                min_dist = dist
                if is_match:
                    best_match = temp.user
                    
        if best_match:
            return {
                "status": "matched",
                "user": best_match.name,
                "identity": best_match.identity_number,
                "distance": f"{min_dist:.4f}"
            }
        else:
            return {"status": "unknown", "distance": f"{min_dist:.4f}"}
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users")
def list_users(session: Session = Depends(get_session)):
    users = session.exec(select(User)).all()
    return users
