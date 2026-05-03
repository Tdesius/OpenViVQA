# OpenViVQA — Vietnamese Visual Question Answering

Hệ thống trả lời câu hỏi dựa trên hình ảnh (Visual Question Answering - VQA) cho các hình ảnh thực tế tại Việt Nam (open-domain). Dự án nghiên cứu và so sánh 5 kiến trúc mô hình khác nhau, từ các kiến trúc Modular truyền thống đến các mô hình hội tụ đa phương thức (Multimodal Foundation Models) hiện đại nhất.

## 🤖 Danh sách mô hình

| Model | Kiến trúc | Ghi chú |
|-------|-----------|---------|
| **A1** | ResNet50 + PhoBERT + LSTM Decoder | Baseline sử dụng cơ chế Attention Pooling để nén đặc trưng. |
| **A2** | ResNet50 + PhoBERT + Transformer Decoder | Cải tiến baseline bằng cách cho phép Decoder tương tác với toàn bộ token ảnh/chữ. |
| **B1** | Qwen2-VL-2B-Instruct (Zero-shot) | Sử dụng trực tiếp sức mạnh của mô hình Pretrained lớn mà không cần huấn luyện. |
| **B2 (SFT)** | Qwen2-VL-2B + QLoRA | Tinh chỉnh (Fine-tuning) mô hình Qwen2-VL trên tập dữ liệu OpenViVQA. |
| **B2 (DPO)** | Qwen2-VL-2B + RLHF | Tối ưu hóa mô hình sau SFT bằng kỹ thuật Direct Preference Optimization (DPO). |

## 📊 Kết quả thực nghiệm

Kết quả được đánh giá trên tập kiểm thử OpenViVQA với các độ đo chính:

| Mô hình | Accuracy (%) | BLEU | ROUGE-L | BERTScore (F1) |
| :--- | :---: | :---: | :---: | :---: |
| **A1 (CNN-LSTM)** | 1.00 | 12.27 | 37.89 | 77.68 |
| **A2 (Transformer)** | 6.00 | 19.14 | 43.91 | 80.36 |
| **B1 (Zero-shot)** | 0.00 | 13.35 | 39.36 | 73.43 |

*Lưu ý: Kết quả trên được trích xuất từ file `Evaluation_All_Models.ipynb`. Mô hình A2 cho thấy sự vượt trội rõ rệt so với A1 nhờ cơ chế Attention linh hoạt của Transformer.*

## 📁 Cấu trúc dự án
```text
├── app.py                      # Giao diện Demo (Gradio) tích hợp cả 5 mô hình
├── vocab.json                  # Từ điển câu trả lời cho mô hình A1, A2
├── notebooks/
│   ├── a1-a2.ipynb             # Huấn luyện mô hình Modular (A1 & A2)
│   ├── ZeroShot_B1_Model.ipynb  # Đánh giá khả năng Zero-shot (B1)
│   ├── Fintuning_B2_model.ipynb # Tinh chỉnh mô hình bằng QLoRA (B2 SFT)
│   ├── RLHF_DPO_Finetuning.ipynb# Tối ưu hóa bằng DPO (B2 DPO)
│   └── Evaluation_All_Models.ipynb # Đánh giá tổng hợp và so sánh chéo
├── A1_A2_model/                # Lưu trữ trọng số mô hình A1, A2 (.pth)
├── qwen2vl_finetuned_final/    # Trọng số tinh chỉnh (LoRA Adapter) của B2 SFT
├── qwen2vl_dpo_final/          # Trọng số tối ưu hóa (LoRA Adapter) của B2 DPO
└── data/openvivqa/             # Dữ liệu hình ảnh và câu hỏi (tự động tải)
```

## 📦 Dữ liệu (Dataset)

Dự án sử dụng bộ dữ liệu **OpenViVQA** (Open Vietnamese Visual Question Answering) được công bố bởi UIT-NLP tại VLSP 2023. Đây là tập dữ liệu VQA quy mô lớn cho tiếng Việt với hình ảnh thực tế đa dạng trong nhiều ngữ cảnh đời sống tại Việt Nam (open-domain).

* **Quy mô dữ liệu**:
    * **Huấn luyện (Train)**: 30,833 cặp câu hỏi - đáp.
    * **Kiểm định (Val)**: 3,545 cặp câu hỏi - đáp.
    * **Kiểm thử (Test)**: 14,035 cặp câu hỏi - đáp[cite: 1].
* **Đặc điểm**: Mỗi hình ảnh đi kèm với ít nhất 3 câu hỏi liên quan đến các thuộc tính như số lượng, màu sắc, vị trí, hoặc nhận diện vật thể.
* **Tải dữ liệu**: Dữ liệu được cấu hình để tự động tải từ HuggingFace Hub thông qua các notebook huấn luyện[cite: 1].
    * 🤗 HuggingFace: [uitnlp/OpenViVQA-dataset](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset)

## 🚀 Cài đặt

Cài đặt các thư viện cần thiết để chạy huấn luyện và giao diện demo:
```bash
pip install torch torchvision transformers peft accelerate bitsandbytes \
            gradio evaluate bert-score rouge-score nltk huggingface_hub
```
## 🖥️ Chạy Demo

Giao diện người dùng được xây dựng bằng **Gradio**, hỗ trợ:
- So sánh kết quả trực quan giữa 5 mô hình
- Đo lường thời gian xử lý

### ▶️ Cách chạy

```bash
python app.py
```

Sau khi khởi chạy, truy cập:

```
http://localhost:7860
```

### ✨ Tính năng

Tại giao diện, bạn có thể:
- Tải lên một hình ảnh (JPG, PNG)
- Nhập câu hỏi tiếng Việt liên quan đến ảnh
- Chọn mô hình:
  - A1
  - A2
  - B1
  - B2 SFT
  - B2 DPO
- Xem câu trả lời + thời gian suy luận

---

## 🔁 Quy trình thực hiện

### 1. Chuẩn bị từ điển
- Chạy các cell đầu trong `a1-a2.ipynb`
- Sinh file `vocab.json` cho mô hình A1, A2

---

### 2. Huấn luyện Modular (Nhóm A)

Notebook: `a1-a2.ipynb`

- A1: ResNet50 + PhoBERT + LSTM
- A2: ResNet50 + PhoBERT + Transformer

---

### 3. Đánh giá Zero-shot (B1)

Notebook: `ZeroShot_B1_Model.ipynb`

- Sử dụng trực tiếp Qwen2-VL
- Không cần huấn luyện

---

### 4. Tinh chỉnh VLM (B2 SFT)

Notebook: `Fintuning_B2_model.ipynb`

- Kỹ thuật: **QLoRA**
- Mục tiêu: thích nghi dữ liệu tiếng Việt

---

### 5. Tối ưu hóa Alignment (B2 DPO)

Notebook: `RLHF_DPO_Finetuning.ipynb`

- Áp dụng **Direct Preference Optimization (DPO)**
- Cải thiện:
  - Độ chính xác
  - Độ ngắn gọn
  - Tính tự nhiên của câu trả lời

---

### 6. Đánh giá tổng hợp

Notebook: `Evaluation_All_Models.ipynb`

- Tính các metrics:
  - Accuracy
  - BLEU
  - ROUGE-L
  - BERTScore
- So sánh định lượng & định tính

---

## 👥 Tác giả (Authors)

Dự án được thực hiện bởi sinh viên  
**Khoa Công nghệ Thông tin – Đại học Tôn Đức Thắng (TDTU)**

- [Tên Thành Viên 1] – MSSV: [MSSV]
- [Tên Thành Viên 2] – MSSV: [MSSV]
- [Tên Thành Viên 3] – MSSV: [MSSV]

---

## 📝 Bản quyền

Dự án phục vụ mục đích **học thuật** trong môn *Deep Learning*.

- Dataset: UIT-NLP (VLSP 2023)
- PhoBERT: VinAI
- Qwen2-VL: Alibaba Qwen Team

Mọi bản quyền thuộc về các đơn vị sở hữu tương ứng.
