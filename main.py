from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from model import infer , interpreter
from PIL import Image
import io


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

@app.get("/")
def read_root():
    return {"Hello":"world"}

@app.post("/img/")
async def read_img(file:UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        output_img = infer(interpreter,image)
        img_io = io.BytesIO()
        output_img.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        return StreamingResponse(img_io, media_type='image/jpeg')
    except Exception as e:
        return {"message": e.args}  