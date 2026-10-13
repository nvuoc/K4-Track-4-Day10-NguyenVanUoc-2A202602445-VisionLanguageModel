# BÁO CÁO CÁC BẪY KỸ THUẬT VÀ LƯU Ý KHI THỰC HIỆN LAB DAY 10

Học viên: **Nguyễn Văn Ước** (Mã SV: **2A202602445**)  
Mô hình: **Qwen/Qwen3-VL-4B-Instruct**  
Môi trường mục tiêu: **Google Colab T4 (16GB VRAM) / Kaggle GPU**  

---

## 1. Bẫy Thang Đo Toạ Độ (Coordinate Scale Trap: 0–1000 vs Pixel)
- **Vấn đề**: Quy ước grounding của dòng mô hình Qwen-VL sử dụng thang đo chuẩn hoá nguyên từ $0$ đến $1000$ đại diện cho tỷ lệ phần nghìn chiều dài/rộng ảnh.
- **Bẫy thường gặp**: Nếu lập trình viên xuất toạ độ theo kích thước pixel tuyệt đối (ví dụ $x_1=256, y_1=144$ cho ảnh 640×360) mà chấm bằng `--coord norm1000`, IoU sẽ bằng $0.0$, dẫn đến F1 = 0 dù mô hình khoanh vùng rất chuẩn.
- **Cách xử lý triệt để**:
  - Trong hàm `yolo_to_box`: chuyển $(xc, yc, w, h) \in [0, 1]$ sang $[0, 1000]$ bằng công thức:
    $x_1 = \text{round}((xc - w/2) \times 1000)$, $y_1 = \text{round}((yc - h/2) \times 1000)$, kẹp trong $[0, 1000]$.
  - Trong prompt: nói rõ ràng *"normalized on a 0-1000 scale"*.
  - Nếu thực hiện thí nghiệm E4 dạng pixel: phải đổi cả prompt, nhãn huấn luyện và truyền `--coord pixel` vào `scripts/score.py`.

---

## 2. Bẫy Lưu Trọng Số Projector Trong Thư Viện PEFT (Projector State Dict Trap)
- **Vấn đề**: Khi sử dụng thư viện `peft`, lệnh `model.save_pretrained("work/adapter")` chỉ lưu các ma trận thích ứng LoRA (adapter weights). Khối Projector (`visual.merger` và `visual.deepstack_merger_list`) là các module gốc của mô hình được gán `requires_grad=True`.
- **Hậu quả**: Khi sang bước suy luận, nếu người làm bài chỉ nạp lại adapter LoRA mà không nạp lại trọng số Projector, mô hình sẽ dùng lại Projector ban đầu của Qwen gốc. Kết quả là điểm F1 của E2 và E3 bị sụt giảm nặng nề và giống hệt như E1!
- **Cách xử lý triệt để**:
  - Khi lưu checkpoint sau khi train:
    ```python
    if setup_name in ["E2", "E3"]:
        projector_weights = {k: v.cpu() for k, v in model.named_parameters() 
                             if "visual.merger" in k or "visual.deepstack_merger_list" in k}
        torch.save(projector_weights, f"work/{setup_name}_projector.pt")
    ```
  - Khi suy luận: nạp adapter trước, sau đó nạp đè state dict của projector:
    ```python
    if projector_path and os.path.exists(projector_path):
        state = torch.load(projector_path, map_location="cuda")
        model.load_state_dict(state, strict=False)
    ```

---

## 3. Bẫy Lượng Tử Hoá Khối Thị Giác (ViT Quantization Trap)
- **Vấn đề**: Mô hình được nạp qua `BitsAndBytesConfig(load_in_4bit=True)`.
- **Nguy cơ**: Nếu không cấu hình `llm_int8_skip_modules=["visual", "lm_head"]`, toàn bộ ViT và Projector sẽ bị ép thành tensor 4-bit `Linear4bit`.
- **Hậu quả**: Các tensor 4-bit không thể mở `requires_grad=True` để huấn luyện gradient ở E2 và E3, khiến code báo lỗi `RuntimeError: only Tensors of floating point dtype can require gradients`.
- **Cách xử lý**: Luôn luôn truyền `llm_int8_skip_modules=["visual", "lm_head"]` khi nạp mô hình.

---

## 4. Bẫy Tràn Số Khi Huấn Luyện Projector (Numerical Overflow & Float32)
- **Vấn đề**: Projector của Qwen3-VL chứa DeepStack Merger kết nối các tầng 5, 11, 17 của ViT.
- **Hậu quả**: Khi tích luỹ gradient qua nhiều tầng trung gian ViT với định dạng fp16, gradient rất dễ vượt quá dải giá trị hạn hẹp của fp16 ($> 65,504$), dẫn đến `Loss = nan` hoặc `inf`.
- **Cách xử lý**: Ép kiểu toàn bộ tham số của Projector sang `torch.float32` trước khi đưa vào optimizer:
  ```python
  for name, param in peft_model.named_parameters():
      if "visual.merger" in name or "visual.deepstack_merger_list" in name:
          param.data = param.data.to(torch.float32)
          param.requires_grad = True
  ```

---

## 5. Bẫy Mask Loss & Quản Lý VRAM Trên Colab T4
- **Mask Loss**: Tuyệt đối phải gán `labels[:, :n_prompt] = -100` để chỉ tính loss trên các token câu trả lời của Assistant. Nếu không, mô hình dành 85% năng lực chỉ để học lại 220 visual tokens của ảnh.
- **Bộ nhớ VRAM**:
  - Với Colab T4 16GB, bắt buộc bật `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`.
  - Đặt `model.config.use_cache = False`.
  - Sử dụng cấu hình `batch_size=2`, `gradient_accumulation_steps=4` (hoặc `batch_size=1`, `grad_accum=8` nếu bật thêm LoRA ViT ở E3) để tránh `CUDA Out Of Memory`.
  - Luôn dọn dẹp bộ nhớ đệm `torch.cuda.empty_cache()` và `gc.collect()` giữa các thí nghiệm.

---

## 6. Bẫy Rò Rỉ Dữ Liệu Theo Frame Video (Temporal Data Leakage)
- **Bản chất**: Dataset gồm các frame video liên tiếp cắt ra từ camera hành trình. Hai frame cách nhau 0.5s gần như giống hệt nhau.
- **Nguy cơ**: Chia ngẫu nhiên (random splitting) sẽ đưa frame $t$ vào train và frame $t+1$ vào test, tạo ra F1 cao giả tạo do học vẹt bối cảnh xe cộ/vỉa hè.
- **Cách xử lý**: Luôn gom toàn bộ frame theo session (ngày giờ ghi hình) bằng `prepare_data.py` với thuật toán băm tất định (deterministic splitting) cố định `seed="day10"`.
