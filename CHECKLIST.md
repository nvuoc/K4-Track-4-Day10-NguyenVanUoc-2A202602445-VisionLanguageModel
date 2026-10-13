# CHECKLIST TIẾN ĐỘ THỰC HIỆN LAB DAY 10

Học viên: **Nguyễn Văn Ước**  
Mã SV: **2A202602445**  
Repository: `K4-Track-4-Day10-NguyenVanUoc-2A202602445-VisionLanguageModel`  
Dataset Version: `d10-21e4070301`  

---

## 1. Môi trường & Bộ kiểm thử ban đầu
- [x] Thiết lập Google Colab T4 / Kaggle GPU môi trường tương thích.
- [x] Cài đặt các gói thư viện đúng phiên bản: `transformers==5.19.0`, `peft==0.21.2`, `trl==1.9.2`, `accelerate==1.15.0`, `bitsandbytes==0.50.2`.
- [x] Chạy bộ test hệ thống ban đầu: `python -m pytest -q` đạt `68 passed, 1 skipped`.

## 2. Chuẩn bị dữ liệu & Chia Session
- [x] Chạy `prepare_data.py` gom nhóm theo session (ngày giờ dashcam), chống rò rỉ dữ liệu giữa các frame liên kề.
- [x] Tạo đủ `train.jsonl` (1,422 ảnh), `val.jsonl` (276 ảnh), `test.jsonl` (311 ảnh) và `meta.json`.
- [x] Trực quan hoá 6 mẫu nhãn gốc kiểm tra tính chính xác.

## 3. Chuyển đổi định dạng nhãn & Thang đo 0–1000
- [x] Viết hàm `yolo_to_box(label, yolo)` ánh xạ toạ độ tâm-kích thước sang [0, 1000].
- [x] Viết hàm `objects_to_answer(objects)` sắp xếp box tất định theo thứ tự (y1, x1) từ trên xuống dưới, trái sang phải.
- [x] Soạn thảo Prompt chuẩn cho mô hình grounding Qwen3-VL.
- [x] Kiểm thử ánh xạ ngược (inverse mapping) khớp 100% toạ độ gốc.

## 4. Thí nghiệm E0 (Zero-shot Baseline)
- [x] Nạp mô hình `Qwen/Qwen3-VL-4B-Instruct` ở chế độ QLoRA NF4 4-bit, bảo lưu `visual` và `lm_head` ở fp16 (`llm_int8_skip_modules=["visual", "lm_head"]`).
- [x] Suy luận trên 311 ảnh tập test, xuất file `work/e0.jsonl`.
- [x] Đánh giá bằng `scripts/score.py`, sinh bằng chứng `report/evidence/e0.json`.
- [x] Phân tích nguyên nhân F1 thấp (< 0.05) và `parse_valid_rate` thấp của model gốc.

## 5. Xây dựng Dữ liệu Huấn luyện & Mask Loss
- [x] Hiện thực `encode_example()` sử dụng Chat Template của Qwen.
- [x] Xác thực tiền tố prompt và gán nhãn `-100` cho toàn bộ prompt và visual tokens (chỉ tính loss trên câu trả lời của Assistant).
- [x] Hiện thực `collate_fn()` đệm token văn bản và nối liên tiếp tensor ảnh (`pixel_values`, `image_grid_thw`).
- [x] Kiểm tra số lượng visual tokens đúng 220 tokens cho ảnh 640×360.

## 6. Thiết lập LoRA & Các Thí nghiệm Finetuning (E1, E2, E3)
- [x] Hiện thực `apply_setup(model, setup_name)`.
- [x] **E1**: LoRA LLM (252 modules, r=16, alpha=32). ViT và Projector đóng băng.
- [x] **E2**: E1 + mở toàn bộ trọng số Projector (109M tham số) đưa về `torch.float32`.
- [x] **E3**: E2 + bổ sung LoRA trên các khối ViT (thêm 96 modules).
- [x] Vòng lặp huấn luyện chuẩn 300 bước, Cosine scheduler 5% warmup, tích luỹ gradient (effective batch size 8).
- [x] Lưu adapter LoRA và lưu riêng file `.pt` cho projector weights của E2/E3.
- [x] Suy luận và chấm điểm sinh đủ `report/evidence/e1.json`, `e2.json`, `e3.json` kèm metadata đầy đủ.

## 7. Thí nghiệm Phân rã Đơn biến E4 (Ablation Study)
- [x] Lựa chọn biến số LoRA Rank ($r=4$ so với $r=64$, giữ nguyên các siêu tham số khác).
- [x] Chạy huấn luyện và đánh giá trên tập test, sinh `report/evidence/e4_rank_4.json` và `e4_rank_64.json`.
- [x] Phân tích giao thoa khoảng tin cậy CI95 bootstrap.

## 8. Thử thách Thưởng E5 (GRPO với Verifiable IoU Reward)
- [x] Viết hàm `compute_reward(completion, gt_boxes)` thưởng IoU và phạt box thừa qua mẫu số.
- [x] Viết bộ unit test đạt 5/5 ca kiểm thử.
- [x] Cấu hình `trl.GRPOTrainer` và chạy tối ưu RL, xuất file `report/evidence/e5.json`.

## 9. Đóng gói & Báo cáo
- [x] Hoàn thiện `REPORT.md` không còn placeholder `<điền>`, trả lời đầy đủ 10 câu hỏi với số liệu thực nghiệm.
- [x] Đính kèm 6 ảnh minh hoạ trực quan hoá box tại `report/images/`.
- [x] Chạy `scripts/check_submission.py` xác nhận đạt **0 FAIL, 0 WARN**.
- [x] Repository sạch sẽ: không commit `data/`, `work/`, `.env`, trọng số mô hình.
- [x] Thực hiện tối thiểu 5 commit Git phân chia theo từng bước thực hiện.
