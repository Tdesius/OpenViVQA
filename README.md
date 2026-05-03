# 🇻🇳 OpenViVQA: Vietnamese Visual Question Answering System

## Overview

**OpenViVQA** là hệ thống **Visual Question Answering (VQA)** cho tiếng Việt trong miền **[Vietnamese Food Domain]**.
Hệ thống nhận **ảnh + câu hỏi tiếng Việt** và sinh ra **câu trả lời ngắn (≤ 10 từ)**.

Project này được xây dựng cho môn **Deep Learning (Final Project)**, kết hợp:

* Computer Vision (CNN)
* NLP (PhoBERT)
* Transformer
* Multimodal Learning
* Reinforcement Learning (DPO / RLHF)

---

## Objectives

* Xây dựng hệ thống VQA tiếng Việt cho miền chuyên biệt
* So sánh các kiến trúc:

  * LSTM vs Transformer decoder
  * Zero-shot vs Fine-tuning
* Cải thiện chất lượng bằng RL (DPO)

---

## Dataset

### Thống kê

* ~2000 samples train (≥200 ảnh)
* ≥50 samples test (ảnh không trùng train)
* Mỗi ảnh ≥3 câu hỏi
* Chia dữ liệu: **80/10/10 (train/val/test)**

### Loại câu hỏi

* Yes/No
* Đếm số lượng
* Nhận dạng (What is this?)
* Thuộc tính (màu sắc, đặc điểm)
* Không gian

### Data Augmentation

* Ảnh: flip, crop, rotate
* Text: paraphrase, back-translation

---

## Model Architectures

### Hướng A — Modular Architecture

#### A1 — CNN + PhoBERT + LSTM Decoder

* Image Encoder: ResNet50 (pretrained)
* Text Encoder: PhoBERT
* Fusion: Concatenation + Attention Pooling
* Decoder: LSTM

#### A2 — CNN + PhoBERT + Transformer Decoder

* Same encoder as A1
* Decoder: Transformer

-> Mục tiêu: so sánh **LSTM vs Transformer**

---

### Hướng B — Multimodal Pretrained

#### B1 — Zero-shot

* Model: Qwen2-VL-2B-Instruct
* Không fine-tune
* Dùng trực tiếp cho tiếng Việt

#### B2 — Fine-tuned (SFT)

* Qwen2-VL + QLoRA
* Train trên dataset OpenViVQA

#### B2 (DPO) — RLHF

* Fine-tune tiếp bằng **Direct Preference Optimization**
* Sử dụng preference data ≥100 cặp

---

## Model Configurations

| Config   | Description                 |
| -------- | --------------------------- |
| A1       | CNN + PhoBERT + LSTM        |
| A2       | CNN + PhoBERT + Transformer |
| B1       | Qwen2-VL Zero-shot          |
| B2       | Qwen2-VL Fine-tuned         |
| B2 (DPO) | Qwen2-VL + RLHF             |

---

## Evaluation Metrics

* **VQA Accuracy**

  * Exact Match
  * Soft Accuracy (VQA v2)
* **BLEU / ROUGE-L / METEOR**
* **BERTScore**
* **LLM-as-a-judge**

---

## Experimental Results (Sample)

### Zero-shot (B1) — Example Predictions

| Question                                   | Ground Truth        | Prediction                |
| ------------------------------------------ | ------------------- | ------------------------- |
| Có bao nhiêu người đi vào chợ bằng xe máy? | Có hai người        | Xe máy chiếm ~30% tổng số |
| Chương trình đang chiếu là gì?             | Thời sự             | "Thời sự" trên VTV4       |
| Đây là gì?                                 | Phòng giao dịch PVI | Bảng hiệu PVI Tây Nam     |
| Có bao nhiêu người trong cửa hàng?         | Một người           | Khoảng 3 người            |
| Dòng chữ Stanley màu gì?                   | Xanh dương          | Xanh lá                   |

Nhận xét:

* Zero-shot hiểu ngữ nghĩa tốt
* Nhưng sai ở:

  * **đếm số lượng**
  * **thuộc tính (màu sắc)**
  * **trả lời dài hơn yêu cầu**

---

## Demo

Chạy ứng dụng:

```bash
python app.py
```

Sau đó mở:

```
http://localhost:7860
```

### 🎮 Demo hỗ trợ:

* Upload ảnh
* Nhập câu hỏi tiếng Việt
* Chọn model:

  * A1 / A2 / B1 / B2 / B2 (DPO)

---

## 📁 Project Structure

```
├── data/
├── A1_A2_model/
├── qwen2vl_finetuned_final/
├── qwen2vl_dpo_final/
├── notebooks/
│   ├── a1-a2.ipynb
│   ├── Fintuning_B2_model.ipynb
│   ├── RLHF_DPO_Finetuning.ipynb
│   ├── Evaluation_All_Models.ipynb
├── app.py
├── vocab.json
└── README.md
```

---

## Key Findings

* Transformer decoder (A2) > LSTM (A1)
* Fine-tuning (B2) >> Zero-shot (B1)
* RLHF (DPO) giúp:

  * trả lời ngắn hơn
  * sát ground truth hơn

---

## Future Work

* Improve counting reasoning
* Better Vietnamese instruction tuning
* Larger dataset
* Try BLIP-2 / LLaVA / PaliGemma

---

## Authors

* [Your Name / Team]

---

## References

* VQA v2 Dataset
* Qwen2-VL
* PhoBERT
* HuggingFace Transformers

---

## Notes

Project phục vụ mục đích học tập trong môn **Deep Learning**.
