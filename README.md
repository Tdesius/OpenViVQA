# OpenViVQA — Vietnamese Visual Question Answering

A research project benchmarking five VQA architectures on the **OpenViVQA** dataset — from traditional modular encoder-decoder models to modern multimodal foundation models fine-tuned on Vietnamese data.

> 📄 Built as a final project for the Deep Learning course at Ton Duc Thang University (TDTU), 2026.

---

## Results

Evaluated on the OpenViVQA test set (14,035 QA pairs):

| Model | Architecture | Accuracy (%) | BLEU | ROUGE-L | METEOR | BERTScore (F1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A1** | ResNet-50 + PhoBERT + LSTM | 1.00 | 12.27 | 37.89 | — | 77.68 |
| **A2** | ResNet-50 + PhoBERT + Transformer | 6.00 | 19.14 | 43.91 | — | 80.36 |
| **B1** | Qwen2-VL-2B (Zero-shot) | 0.00 | 13.35 | 39.36 | — | 73.43 |
| **B2 (SFT)** | Qwen2-VL-2B + QLoRA | **14.00** | **33.40** | **61.80** | **59.10** | **85.60** |
| **B2 (DPO)** | Qwen2-VL-2B + QLoRA + DPO | 0.00 | 6.00 | 33.70 | 25.20 | — |

**Key findings:**
- B2 (SFT) outperforms the from-scratch Transformer baseline (A2) by **+18 ROUGE-L points** and **+5.2 BERTScore points**
- B1 zero-shot accuracy is 0% due to language/format mismatch with Vietnamese ground-truth, but semantic scores (ROUGE-L 39.4, BERTScore 73.4) show the model understands the content
- DPO degraded performance on all metrics — attributed to small preference dataset (190 pairs) and format drift from exact-match ground-truth

---

## Models

| ID | Architecture | Notes |
|---|---|---|
| **A1** | ResNet-50 + PhoBERT + LSTM Decoder | Baseline; uses attention pooling to compress visual+text features into a single vector |
| **A2** | ResNet-50 + PhoBERT + Transformer Decoder | Replaces LSTM with multi-head cross-attention; decoder attends directly to all image/text tokens |
| **B1** | Qwen2-VL-2B-Instruct (Zero-shot) | No fine-tuning; tests raw capability of a pretrained multimodal LLM on Vietnamese |
| **B2 (SFT)** | Qwen2-VL-2B + QLoRA | Fine-tuned on 30,833 Vietnamese QA pairs; 1.64% trainable parameters via LoRA |
| **B2 (DPO)** | Qwen2-VL-2B + QLoRA + DPO | Post-SFT alignment using Direct Preference Optimization on 190 preference pairs |

---

## Dataset

**OpenViVQA** — published by UIT-NLP at VLSP 2023. Large-scale open-domain VQA dataset for Vietnamese with real-world images covering everyday contexts in Vietnam.

| Split | QA Pairs |
|---|---|
| Train | 30,833 |
| Validation | 3,545 |
| Test | 14,035 |

Each image comes with at least 3 questions covering attributes like quantity, color, position, and object recognition.

- 🤗 HuggingFace: [uitnlp/OpenViVQA-dataset](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset)
- 📦 Kaggle: [windyy261203/openvivqa](https://www.kaggle.com/datasets/windyy261203/openvivqa)

---

## Project Structure

```
├── app.py                          # Gradio demo — compare all 5 models side by side
├── vocab.json                      # Answer vocabulary for A1, A2
├── requirements.txt
├── notebooks/
│   ├── A1_A2.ipynb                 # Train modular models (A1 & A2)
│   ├── ZeroShot_B1_Model.ipynb     # Zero-shot evaluation (B1)
│   ├── Fintuning_B2_model.ipynb    # QLoRA fine-tuning (B2 SFT)
│   ├── RLHF_DPO_Finetuning.ipynb  # DPO alignment (B2 DPO)
│   └── Evaluation_All_Models.ipynb # Cross-model evaluation & metrics
└── data/openvivqa/                 # Auto-downloaded from HuggingFace
```

---

## Pretrained Weights

Model weights are hosted on HuggingFace due to file size:

🔗 **[HarryT1211/DL_VQA](https://huggingface.co/HarryT1211/DL_VQA/tree/main)**

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="HarryT1211/DL_VQA", local_dir="./A1_A2_model")
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Demo

Built with **Gradio** — compare all five models on any image and Vietnamese question.

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

**Features:**
- Upload any image (JPG, PNG)
- Type a Vietnamese question
- Select one or more models (A1, A2, B1, B2 SFT, B2 DPO)
- See the answer + inference time per model

---

## Pipeline

| Step | Notebook | Description |
|---|---|---|
| 1 | `A1_A2.ipynb` | Generate `vocab.json`, train A1 and A2 |
| 2 | `ZeroShot_B1_Model.ipynb` | Zero-shot inference with Qwen2-VL |
| 3 | `Fintuning_B2_model.ipynb` | QLoRA fine-tuning on OpenViVQA |
| 4 | `RLHF_DPO_Finetuning.ipynb` | DPO alignment post-SFT |
| 5 | `Evaluation_All_Models.ipynb` | Compute Accuracy, BLEU, ROUGE-L, METEOR, BERTScore |

---

## Authors

Students at the **Faculty of Information Technology, Ton Duc Thang University (TDTU)**

| Name | Student ID |
|---|---|
| Pham Quoc Hung | 523H0135 |
| Dinh Bui Khanh Huy | 523H0136 |
| Nguyen Dong Quan | 523H0171 |

---

## Credits

- Dataset: [UIT-NLP](https://github.com/uitnlp) (VLSP 2023)
- PhoBERT: [VinAI Research](https://github.com/VinAIResearch/PhoBERT)
- Qwen2-VL: [Alibaba Qwen Team](https://github.com/QwenLM/Qwen2-VL)

*This project is for academic purposes under the Deep Learning course.*

---

<details>
<summary>📖 Đọc bằng tiếng Việt</summary>

## Giới thiệu

Hệ thống trả lời câu hỏi dựa trên hình ảnh (Visual Question Answering - VQA) cho các hình ảnh thực tế tại Việt Nam (open-domain). Dự án nghiên cứu và so sánh 5 kiến trúc mô hình khác nhau, từ các kiến trúc Modular truyền thống đến các mô hình hội tụ đa phương thức (Multimodal Foundation Models) hiện đại nhất.

## Kết quả

| Mô hình | Accuracy (%) | BLEU | ROUGE-L | METEOR | BERTScore (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A1 (CNN-LSTM) | 1.00 | 12.27 | 37.89 | — | 77.68 |
| A2 (Transformer) | 6.00 | 19.14 | 43.91 | — | 80.36 |
| B1 (Zero-shot) | 0.00 | 13.35 | 39.36 | — | 73.43 |
| B2 (SFT) | **14.00** | **33.40** | **61.80** | **59.10** | **85.60** |
| B2 (DPO) | 0.00 | 6.00 | 33.70 | 25.20 | — |

## Cài đặt và chạy demo

```bash
pip install -r requirements.txt
python app.py
```

Truy cập `http://localhost:7860` để sử dụng giao diện Gradio so sánh 5 mô hình.

## Tác giả

Dự án được thực hiện bởi sinh viên Khoa Công nghệ Thông tin – Đại học Tôn Đức Thắng (TDTU):

- Phạm Quốc Hưng – 523H0135
- Đinh Bùi Khánh Huy – 523H0136
- Nguyễn Đông Quân – 523H0171

</details>
