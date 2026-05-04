"""
app.py — Vietnamese Food VQA Demo
Hỗ trợ 4 mô hình: A1, A2, B1 (zero-shot), B2 (SFT), B2-DPO
"""

import os, json, time, warnings
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import gradio as gr
from transformers import AutoModel, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

warnings.filterwarnings("ignore")

CKPT_A1        = "A1_A2_model/best_a1.pth"
CKPT_A2        = "A1_A2_model/best_a2.pth"
QWEN_BASE_ID   = "Qwen/Qwen2-VL-2B-Instruct"
SFT_ADAPTER    = "./qwen2vl_finetuned_final"
DPO_ADAPTER    = "./qwen2vl_dpo_final"
VOCAB_PATH     = "vocab.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AnswerVocab:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self): return len(self.word2idx)

    def decode(self, ids):
        words = []
        for t in ids:
            if t == 2: break
            if t > 3: words.append(self.idx2word.get(t, ""))
        return " ".join(w for w in words if w)

try:
    vocab = AnswerVocab(VOCAB_PATH)
    VOCAB_SIZE = len(vocab)
    print(f"Vocab size: {VOCAB_SIZE}")
except Exception as e:
    vocab = None
    VOCAB_SIZE = 6421
    print(f"Cảnh báo: Không đọc được vocab.json ({e}) — dùng vocab_size mặc định {VOCAB_SIZE}")

# KIẾN TRÚC A1 / A2  (khớp chính xác với a1-a2.ipynb)
class ImageEncoder(nn.Module):
    def __init__(self, embed_size=512, finetune_layers=1):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone = list(resnet.children())[:-2]
        self.frozen   = nn.Sequential(*backbone[:-finetune_layers])
        self.finetune = nn.Sequential(*backbone[-finetune_layers:])
        for p in self.frozen.parameters(): p.requires_grad = False
        self.proj = nn.Linear(2048, embed_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        with torch.no_grad(): x = self.frozen(x)
        x = self.finetune(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return self.norm(self.proj(x))

class TextEncoder(nn.Module):
    def __init__(self, embed_size=512, finetune_layers=2):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        for p in self.phobert.parameters(): p.requires_grad = False
        for layer in self.phobert.encoder.layer[-finetune_layers:]:
            for p in layer.parameters(): p.requires_grad = True
        self.proj = nn.Linear(768, embed_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        return self.norm(self.proj(out.last_hidden_state))

class MultimodalFusion(nn.Module):
    def __init__(self, embed_size=512, hidden_size=512):
        super().__init__()
        self.proj = nn.Linear(embed_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, img, txt):
        return self.norm(self.proj(torch.cat([img, txt], dim=1)))

class VQAModelA1(nn.Module):
    def __init__(self, vocab_size, embed_size=512, hidden_size=512,
                 finetune_img=1, finetune_txt=2, dropout=0.3):
        super().__init__()
        self.img_encoder = ImageEncoder(embed_size, finetune_img)
        self.txt_encoder = TextEncoder(embed_size, finetune_txt)
        self.fusion      = MultimodalFusion(embed_size, hidden_size)
        self.attn_pool   = nn.Linear(hidden_size, 1)
        self.embed       = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout     = nn.Dropout(dropout)
        self.lstm        = nn.LSTM(embed_size, hidden_size, num_layers=2,
                                   batch_first=True, dropout=dropout)
        self.fc_out      = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, input_ids, attention_mask, max_len=20):
        img  = self.img_encoder(images)
        txt  = self.txt_encoder(input_ids, attention_mask)
        mem  = self.fusion(img, txt)
        sc   = self.attn_pool(mem).squeeze(-1)
        wt   = torch.softmax(sc, dim=-1).unsqueeze(-1)
        pool = (mem * wt).sum(1)
        h0   = pool.unsqueeze(0).repeat(2, 1, 1)
        c0   = torch.zeros_like(h0)
        B    = images.size(0)
        curr = torch.full((B, 1), 1, dtype=torch.long, device=images.device)
        h, c = h0, c0
        preds = []
        for _ in range(max_len):
            emb = self.dropout(self.embed(curr))
            out, (h, c) = self.lstm(emb, (h, c))
            nxt = self.fc_out(out).argmax(-1)
            preds.append(nxt); curr = nxt
            if (nxt == 2).all(): break
        return torch.cat(preds, 1)

class VQAModelA2(nn.Module):
    def __init__(self, vocab_size, embed_size=512, hidden_size=512,
                 num_heads=8, num_layers=4, dropout=0.2,
                 finetune_img=1, finetune_txt=2):
        super().__init__()
        self.img_encoder = ImageEncoder(embed_size, finetune_img)
        self.txt_encoder = TextEncoder(embed_size, finetune_txt)
        self.fusion      = MultimodalFusion(embed_size, hidden_size)
        self.embed       = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_embed   = nn.Embedding(512, hidden_size)
        self.dropout     = nn.Dropout(dropout)
        dec_layer        = nn.TransformerDecoderLayer(
            hidden_size, num_heads, hidden_size * 4, dropout,
            batch_first=True, norm_first=True)
        self.decoder     = nn.TransformerDecoder(dec_layer, num_layers)
        self.out_norm    = nn.LayerNorm(hidden_size)
        self.fc_out      = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, input_ids, attention_mask, max_len=20):
        img  = self.img_encoder(images)
        txt  = self.txt_encoder(input_ids, attention_mask)
        mem  = self.fusion(img, txt)
        B    = images.size(0)
        seq  = torch.full((B, 1), 1, dtype=torch.long, device=images.device)
        for _ in range(max_len):
            T   = seq.size(1)
            pos = torch.arange(T, device=seq.device).unsqueeze(0)
            tgt = self.dropout(self.embed(seq) + self.pos_embed(pos))
            msk = nn.Transformer.generate_square_subsequent_mask(T, device=seq.device)
            out = self.decoder(tgt, mem, tgt_mask=msk, tgt_key_padding_mask=(seq == 0))
            nxt = self.fc_out(self.out_norm(out))[:, -1, :].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
            if (nxt == 2).all(): break
        return seq[:, 1:]

# LOAD MODELS (lazy — chỉ tải khi cần)
_cache = {}

IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def _load_phobert_tokenizer():
    if "phobert_tok" not in _cache:
        _cache["phobert_tok"] = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    return _cache["phobert_tok"]

def _load_qwen_processor():
    if "qwen_proc" not in _cache:
        _cache["qwen_proc"] = AutoProcessor.from_pretrained(QWEN_BASE_ID, max_pixels=313600)
    return _cache["qwen_proc"]

def _load_checkpoint_state(path):
    """Đọc state_dict từ checkpoint, tự động nhận diện key đúng."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # best_a1/a2.pth dùng key 'model_state'
    # saved_models dùng state_dict thẳng (flat dict)
    for key in ("model_state", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
    return ckpt  # flat state_dict

def _load_a1():
    if "A1" not in _cache:
        model = VQAModelA1(VOCAB_SIZE).to(device)
        if os.path.exists(CKPT_A1):
            state = _load_checkpoint_state(CKPT_A1)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"✓ A1 loaded  |  missing={len(missing)}  unexpected={len(unexpected)}")
        else:
            print(f"⚠ A1 checkpoint not found: {CKPT_A1}")
        model.eval()
        _cache["A1"] = model
    return _cache["A1"]

def _load_a2():
    if "A2" not in _cache:
        model = VQAModelA2(VOCAB_SIZE).to(device)
        if os.path.exists(CKPT_A2):
            state = _load_checkpoint_state(CKPT_A2)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"✓ A2 loaded  |  missing={len(missing)}  unexpected={len(unexpected)}")
        else:
            print(f"⚠ A2 checkpoint not found: {CKPT_A2}")
        model.eval()
        _cache["A2"] = model
    return _cache["A2"]

def _load_b1():
    if "B1" not in _cache:
        print("Đang tải B1 (Qwen Zero-shot)…")
        m = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_BASE_ID,
            device_map={"": str(device)},
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        )
        m.eval()
        _cache["B1"] = m
        print("✓ B1 loaded")
    return _cache["B1"]

def _load_b2():
    if "B2" not in _cache:
        if not os.path.exists(SFT_ADAPTER):
            print(f"⚠ B2 adapter not found: {SFT_ADAPTER}")
            _cache["B2"] = None
        else:
            print("Đang tải B2 (SFT)…")
            base = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_BASE_ID,
                device_map={"": str(device)},
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
            )
            m = PeftModel.from_pretrained(base, SFT_ADAPTER)
            m.eval()
            _cache["B2"] = m
            print("✓ B2 (SFT) loaded")
    return _cache["B2"]

def _load_b2_dpo():
    if "B2_DPO" not in _cache:
        if not os.path.exists(DPO_ADAPTER):
            print(f"⚠ B2-DPO adapter not found: {DPO_ADAPTER}")
            _cache["B2_DPO"] = None
        else:
            print("Đang tải B2-DPO…")
            base = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_BASE_ID,
                device_map={"": str(device)},
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
            )
            m = PeftModel.from_pretrained(base, DPO_ADAPTER)
            m.eval()
            _cache["B2_DPO"] = m
            print("✓ B2 (DPO) loaded")
    return _cache["B2_DPO"]

@torch.no_grad()
def _infer_ab(model_key, image: Image.Image, question: str) -> str:
    model = _load_a1() if model_key == "A1" else _load_a2()
    tok   = _load_phobert_tokenizer()
    if vocab is None:
        return "Lỗi: vocab.json không tồn tại."
    img_t = IMG_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    enc   = tok(question, padding="max_length", max_length=20,
                truncation=True, return_tensors="pt")
    iids  = enc.input_ids.to(device)
    mask  = enc.attention_mask.to(device)
    ids   = model(img_t, iids, mask)
    return vocab.decode(ids[0].cpu().tolist())

@torch.no_grad()
def _infer_qwen(model, image: Image.Image, question: str, max_new_tokens=30) -> str:
    proc  = _load_qwen_processor()
    msgs  = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}
    ]}]
    text  = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp   = proc(text=[text], images=[image.convert("RGB")],
                 padding=True, return_tensors="pt")
    inp   = {k: v.to(model.device) for k, v in inp.items()}
    out   = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inp["input_ids"].shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

def predict(image, question, model_key):
    if image is None:
        return "⚠ Vui lòng tải ảnh lên.", ""
    if not question.strip():
        return "⚠ Vui lòng nhập câu hỏi.", ""

    t0 = time.time()
    try:
        if model_key in ("A1", "A2"):
            answer = _infer_ab(model_key, image, question)
        elif model_key == "B1":
            answer = _infer_qwen(_load_b1(), image, question)
        elif model_key == "B2 (SFT)":
            m = _load_b2()
            answer = _infer_qwen(m, image, question) if m else "B2 chưa được train."
        elif model_key == "B2 (DPO)":
            m = _load_b2_dpo()
            answer = _infer_qwen(m, image, question) if m else "B2-DPO chưa được train."
        else:
            answer = "Mô hình không hợp lệ."
    except Exception as e:
        answer = f"Lỗi inference: {e}"

    elapsed = time.time() - t0
    return answer, f"{elapsed:.2f}s"

MODEL_CHOICES = [
    "A1 (CNN-LSTM)",
    "A2 (Transformer)",
    "B1 (Zero-shot)",
    "B2 (SFT)",
    "B2 (DPO)",
]

DESCRIPTION = """
## Vietnamese VQA — Demo

Hệ thống trả lời câu hỏi dựa trên hình ảnh (**Visual Question Answering**) cho ảnh mở (open-domain) tiếng Việt.  
Hỗ trợ **5 mô hình**:

| Mô hình | Kiến trúc | Ghi chú |
|---------|-----------|---------|
| **A1** | ResNet50 + PhoBERT + LSTM Decoder | Baseline |
| **A2** | ResNet50 + PhoBERT + Transformer Decoder | Cải tiến |
| **B1** | Qwen2-VL-2B (Zero-shot) | Không fine-tune |
| **B2 (SFT)** | Qwen2-VL-2B + QLoRA Fine-tuned | Fine-tune trên OpenViVQA |
| **B2 (DPO)** | Qwen2-VL-2B + QLoRA + DPO | Tối ưu bằng RLHF |
"""

with gr.Blocks(
    title="Vietnamese VQA",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css="""
        .answer-box { font-size: 1.4em; font-weight: bold; color: #1a73e8; }
        .time-box   { font-size: 0.85em; color: #666; }
        footer      { display: none !important; }
    """
) as demo:

    gr.Markdown(DESCRIPTION)
    gr.Divider()

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                type="pil",
                label="📷 Ảnh đầu vào",
                height=320
            )
            question_input = gr.Textbox(
                lines=2,
                placeholder="Ví dụ: Đây là món ăn gì? / Bao nhiêu người trong ảnh?",
                label="❓ Câu hỏi"
            )
            model_choice = gr.Radio(
                choices=MODEL_CHOICES,
                value="B2 (SFT)",
                label="🤖 Chọn mô hình"
            )
            submit_btn = gr.Button("🔍 Dự đoán", variant="primary", size="lg")

        with gr.Column(scale=1):
            answer_out = gr.Textbox(
                label="💬 Câu trả lời",
                lines=3,
                elem_classes=["answer-box"],
                interactive=False
            )
            time_out = gr.Textbox(
                label="⏱ Thời gian xử lý",
                elem_classes=["time-box"],
                interactive=False
            )

            gr.Markdown("### 📸 Ví dụ mẫu")
            gr.Examples(
                examples=[
                    ["data/openvivqa/dev/dev-images/0.jpg", "Đây là gì?", "B2 (SFT)"],
                ],
                inputs=[img_input, question_input, model_choice],
                label="Click để thử"
            )

    submit_btn.click(
        fn=predict,
        inputs=[img_input, question_input, model_choice],
        outputs=[answer_out, time_out]
    )

    gr.Divider()
    gr.Markdown(
        "*Đồ án môn học — Fine-tuning Qwen2-VL cho bài toán VQA tiếng Việt (OpenViVQA).*",
        elem_classes=["time-box"]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)