# OpenViVQA — Vietnamese Visual Question Answering

A Visual Question Answering (VQA) system for open-domain Vietnamese images, built as a course project. Five models are implemented and evaluated, ranging from a custom CNN-LSTM baseline to a fine-tuned multimodal

---

## 📁 Project Structure

```
├── app.py                          # Gradio demo (5 models)
├── vocab.json                      # Answer vocabulary
├── a1-a2.ipynb                     # Train A1 & A2 models
├── ZeroShot_B1_Model.ipynb         # B1 zero-shot evaluation
├── Fintuning_B2_model.ipynb        # B2 SFT fine-tuning (QLoRA)
├── RLHF_DPO_Finetuning.ipynb       # B2 DPO alignment
├── Evaluation_All_Models.ipynb     # Unified evaluation across all models
├── A1_A2_model/
│   ├── best_a1.pth
│   └── best_a2.pth
├── qwen2vl_finetuned_final/        # B2 SFT adapter (LoRA)
├── qwen2vl_dpo_final/              # B2 DPO adapter (LoRA)
└── data/openvivqa/                 # Dataset (auto-downloaded)
```

---

## 🤖 Models

| Model | Architecture | Notes |
|-------|-------------|-------|
| **A1** | ResNet50 + PhoBERT + LSTM Decoder | Baseline |
| **A2** | ResNet50 + PhoBERT + Transformer Decoder | Improved baseline |
| **B1** | Qwen2-VL-2B-Instruct (Zero-shot) | No fine-tuning |
| **B2 (SFT)** | Qwen2-VL-2B + QLoRA | Fine-tuned on OpenViVQA |
| **B2 (DPO)** | Qwen2-VL-2B + QLoRA + DPO | RLHF-aligned |

---

## 📊 Evaluation Metrics

- VQA Accuracy
- BLEU
- ROUGE-L
- METEOR
- BERTScore
- LLM-as-a-Judge (0–10)

Run `Evaluation_All_Models.ipynb` to reproduce all results.

---

## 📦 Dataset

**OpenViVQA** — Open Vietnamese Visual Question Answering dataset by UIT-NLP.

- 🤗 HuggingFace: [uitnlp/OpenViVQA-dataset](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset)
- Paper: [OpenViVQA: Task, Dataset, and Multimodal FusionModels for Visual Question Answering in Vietnamese](https://www.researchgate.net/publication/370604978_OpenViVQA_Task_Dataset_and_Multimodal_Fusion_Models_for_Visual_Question_Answering_in_Vietnamese)]

The dataset is auto-downloaded from HuggingFace Hub when you run the notebooks.

---

## 🚀 Setup

```bash
pip install torch torchvision transformers peft accelerate bitsandbytes \
            gradio evaluate bert-score rouge-score nltk huggingface_hub
```

### Base models used
- **PhoBERT**: [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2)
- **Qwen2-VL**: [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

---

## 🖥️ Demo

```bash
python app.py
# Open http://localhost:7860
```

Upload a Vietnamese image, type a question in Vietnamese, and select a model.

---

## 🔁 Training Pipeline

1. **A1 & A2** — Run `a1-a2.ipynb`
2. **B1** — Run `ZeroShot_B1_Model.ipynb` (no training needed)
3. **B2 SFT** — Run `Fintuning_B2_model.ipynb` (QLoRA on Qwen2-VL-2B)
4. **B2 DPO** — Run `RLHF_DPO_Finetuning.ipynb` (DPO on the SFT adapter)
5. **Evaluate** — Run `Evaluation_All_Models.ipynb`

---

## 📝 License

For academic/course use only.
