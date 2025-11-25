import torch
from fastapi import FastAPI, UploadFile
from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizer
from PIL import Image
import io

# ------------------------------
# 1. 只加载一次模型（常驻内存）
# ------------------------------
MODEL_PATH = "/mnt/pfs/3zpd5q/code/EVA-CLIP-8B"

tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH)
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda().eval()

app = FastAPI()

# ------------------------------
# 2. 推理接口，不再加载模型
# ------------------------------
@app.post("/clip")
async def clip_infer(image: UploadFile, texts: list[str]):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))

    input_pixels = processor(images=img, return_tensors="pt").pixel_values.cuda()
    input_ids    = tokenizer(texts, return_tensors="pt", padding=True).input_ids.cuda()

    with torch.no_grad(), torch.cuda.amp.autocast():
        img_feat  = model.encode_image(input_pixels)
        txt_feat  = model.encode_text(input_ids)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    scores = (img_feat @ txt_feat.T).softmax(dim=-1).cpu().tolist()
    return {"scores": scores}
