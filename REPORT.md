Họ tên: Nguyễn Văn Ước (Mã SV: 2A202602445)
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | - | - | 0.031 | [0.015, 0.048] | 0.59 | 0.043 | 0.026 | 0.030 |
| E1 | LoRA LLM | 28,311,552 | 9.5 | 24.0 | 0.529 | [0.478, 0.577] | 0.65 | 0.573 | 0.470 | 0.603 |
| E2 | + projector | 137,359,360 | 11.2 | 28.5 | 0.575 | [0.525, 0.623] | 0.60 | 0.635 | 0.525 | 0.620 |
| E3 | + LoRA ViT | 156,233,728 | 13.8 | 34.2 | 0.605 | [0.556, 0.652] | 0.58 | 0.667 | 0.559 | 0.642 |
| E4 | rank: r=4 (alpha=8) | 141,724,288 | 12.4 | 30.5 | 0.554 | [0.504, 0.602] | 0.62 | 0.609 | 0.504 | 0.606 |
| E4 | rank: r=64 (alpha=128) | 214,271,360 | 14.6 | 39.8 | 0.614 | [0.565, 0.660] | 0.57 | 0.681 | 0.569 | 0.645 |
| E5 | Bonus GRPO (reward IoU) | 156,233,728 | 14.2 | 18.5 | 0.643 | [0.595, 0.689] | 0.47 | 0.707 | 0.595 | 0.682 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Mô hình Qwen/Qwen3-VL-4B-Instruct có tổng quy mô tham số xấp xỉ 4.44B, phân bổ thành 3 khối kiến trúc chính:
1. **Vision Transformer (ViT)**: Quy mô ≈ 415M tham số. Đảm nhiệm vai trò trích xuất đặc trưng thị giác từ ảnh đầu vào 640×360 qua patchify (14×14) và gộp không gian 2×2, tạo ra đúng 220 visual tokens.
2. **Projector / Feature Merger**: Quy mô ≈ 109,047,808 tham số (≈ 109M), gồm:
   - `visual.merger`: ≈ 27M tham số (ánh xạ output lớp cuối của ViT sang không gian nhúng của LLM).
   - `visual.deepstack_merger_list`: ≈ 82M tham số (kết nối đa tầng DeepStack từ các khối 5, 11, 17 của ViT vào các tầng của LLM).
3. **Language Model Core (LLM)**: Quy mô ≈ 4.0B tham số (36 transformer decoder blocks kèm embedding và lm_head).

Cụ thể từng thí nghiệm cập nhật các khối sau:
- **E1 (LoRA LLM)**: Đóng băng hoàn toàn ViT và Projector. Áp dụng LoRA (r=16, alpha=32) trên 7 ma trận chiếu (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) qua 36 tầng LLM (36 × 7 = 252 modules). Tổng số tham số huấn luyện là **28,311,552** (≈ 28.3M tham số, chiếm 0.64% mô hình).
- **E2 (+ Projector)**: Giữ nguyên LoRA LLM như E1, đồng thời mở toàn bộ trọng số của Projector (`visual.merger` và `visual.deepstack_merger_list`) với `requires_grad=True` ở định dạng `float32`. Tổng tham số huấn luyện là **137,359,360** (tăng đúng 109,047,808 tham số so với E1).
- **E3 (+ LoRA ViT)**: Giữ nguyên cấu hình E2 (LoRA LLM + Projector float32) và mở thêm LoRA trên các khối attention và MLP của ViT (`qkv, proj, linear_fc1, linear_fc2` trên 24 khối ViT, tương ứng 96 modules LoRA). Tổng tham số huấn luyện đạt **156,233,728** (tăng thêm 18,874,368 tham số so với E2).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
Ở thí nghiệm Zero-shot E0 trên 311 ảnh tập test:
- **`parse_valid_rate`** đạt **0.4212** (chỉ có 131/311 ảnh trả về định dạng JSON parse được; 180 ảnh bị loại bỏ hoặc hỏng định dạng). F1 tổng thể chỉ đạt **0.0308** (Precision: 0.0612, Recall: 0.0206).
- **Hai lỗi phổ biến nhất của model gốc Qwen3-VL-4B-Instruct chưa finetune**:
  1. *Lỗi văn phong hội thoại và rào đón (Conversational Chatter & Markdown Padding)*: Thay vì trả về một chuỗi JSON thuần tuý, mô hình gốc có xu hướng chat tự nhiên, chèn thêm câu dẫn nhập như *"Based on the dashcam image, here are the road damages detected:"* hoặc bọc trong code block giải thích dài dòng, dẫn đến parser regex bị lỗi hoặc trích xuất không trọn vẹn.
  2. *Lỗi hallucinative coordinates & sai thang đo*: Mô hình gốc không hiểu ngữ nghĩa thang đo chuẩn hoá 0–1000 cho camera hành trình đường bộ; nó sinh ra các toạ độ pixel tuỳ ý (ví dụ [120, 340, 280, 420] cho ảnh 640x360), hoặc bịa ra các nhãn nằm ngoài tập 3 lớp (như "asphalt", "shadow", "car", "road"), dẫn đến Precision và Recall gần như bằng 0.
- **Ví dụ output thô thực tế từ E0**:
```text
I can see several issues on this road. Here is the JSON breakdown:
```json
[
  {"label": "asphalt patch", "bbox_2d": [140, 220, 310, 290]},
  {"label": "pothole", "bbox_2d": [0.35, 0.62, 0.58, 0.81]}
]
```
Please let me know if you need bounding box annotations in pixel values instead.
```
*Nhận xét*: Output này bị lỗi nặng vì vừa chứa chữ thừa, vừa sinh nhãn ngoài tập ("asphalt patch"), vừa trộn lẫn toạ độ thực [0, 1] thay vì số nguyên [0, 1000].

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Hậu quả khi lệch thang đo**: Nếu mô hình sinh toạ độ theo pixel tuyệt đối (chiều rộng 640, chiều cao 360) nhưng kịch bản chấm điểm mặc định theo thang chuẩn hoá `norm1000` (hoặc ngược lại), điểm **F1 sẽ sụp đổ về 0.000**, dù cho mô hình nhận diện chuẩn xác 100% vị trí vật thể.
- **Nguyên nhân toán học**: Ảnh có kích thước thực là 640 × 360. Một ổ gà nằm ở giữa ảnh có toạ độ pixel xấp xỉ `[256, 144, 384, 216]`. Nếu chấm trên thang 0–1000 mà không quy đổi, ground truth ở thang 1000 là `[400, 400, 600, 600]`. Giao diện giữa hai box này hoàn toàn tách rời (x2_pred = 384 < x1_gt = 400), dẫn đến diện tích giao $Inter = 0 \implies \text{IoU} = 0.0$. Vì ngưỡng IoU yêu cầu là $\ge 0.5$, toàn bộ box dự đoán bị tính là False Positive ($FP$) và mọi box thật thành False Negative ($FN$), khiến $Precision = Recall = F1 = 0$.
- **Kiểm chứng thực tế**: Điều này đã được kiểm chứng bằng test case `test_pixel_vs_norm1000_scale_trap` trong `tests/test_score.py`: cùng một dự đoán pixel chuẩn, khi chấm với cờ `--coord pixel` thì đạt F1 = 1.0 (hoàn hảo), nhưng khi chấm bằng `--coord norm1000` thì F1 rớt thẳng về 0.000 dù `parse_valid_rate` vẫn đạt 1.0.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Lý do chỉ tính loss trên phần trả lời (Assistant Answer)**: Trong bài toán Vision-Language grounding, nhiệm vụ của mô hình là suy luận có điều kiện (conditional generation): cho trước ảnh và câu lệnh, hãy sinh chuỗi JSON toạ độ. Nhãn chỉ được tính Cross-Entropy trên các token của Assistant Answer bằng cách gán `labels[:, :n_prompt] = -100` (giá trị mặc định bỏ qua trong PyTorch CrossEntropyLoss).
- **Hậu quả nếu tính loss trên toàn bộ sequence (cả prompt và 220 visual tokens)**:
  1. *Gradient dissipation (Phân tán gradient)*: Một ảnh 640×360 nén thành 220 visual tokens, cộng thêm prompt chiếm khoảng 60 tokens, tổng cộng phần input chiếm hơn 280 tokens (trên 85% tổng độ dài chuỗi ~330 tokens). Chuỗi JSON đích chỉ dài khoảng 30–50 tokens.
  2. *Lạc hướng mục tiêu tối ưu*: Nếu tính loss trên toàn bộ token, mô hình sẽ dành 85% ngân sách cập nhật gradient chỉ để "tái tạo lại các token biểu diễn ảnh" (image reconstruction) hoặc "học vẹt câu lệnh hệ thống", thay vì học mối tương quan không gian giữa đặc trưng thị giác và toạ độ số 0–1000.
  3. Kết quả là mô hình hội tụ cực kỳ chậm hoặc hoàn toàn không học được cú pháp JSON toạ độ, dẫn đến loss không phản ánh chất lượng phát hiện hư hại.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vấn đề lượng tử hoá 4-bit**: Khi cấu hình BitsAndBytes NF4 mà không loại trừ khối thị giác, toàn bộ ViT (bao gồm `visual.merger` và `visual.deepstack_merger_list`) sẽ bị đóng băng thành các tensor lượng tử hoá 4-bit (`Linear4bit`). Các tensor này không hỗ trợ lan truyền ngược gradient và cập nhật trọng số liên tục.
- **Cách xử lý**: Thiết lập cấu hình lượng tử hoá với tham số bắt buộc:
  `llm_int8_skip_modules = ["visual", "lm_head"]`
  Tham số này bảo vệ toàn bộ nhánh ViT, Projector và đầu ra lm_head ở định dạng dấu phẩy động 16-bit (fp16), cho phép mở cờ `requires_grad=True` và cập nhật gradient đầy đủ ở E2 và E3.
- **Lý do Projector phải ép kiểu sang `torch.float32`**:
  Projector trong Qwen3-VL không chỉ là một tầng chiếu đơn giản mà chứa cả DeepStack Merger kết nối từ các tầng 5, 11, 17 của ViT. Khi tích luỹ và lan truyền gradient qua mạng sâu dưới dạng mixed precision (fp16), miền giá trị của fp16 (khoảng $6 \times 10^{-5}$ đến $65504$) rất hẹp. Việc cộng dồn gradient từ nhiều tầng trung gian ViT vào projector rất dễ gây hiện tượng tràn số (numerical overflow / underflow), khiến gradient biến thành `inf` hoặc `nan` làm sụp đổ toàn bộ quá trình huấn luyện. Ép kiểu projector về `torch.float32` giải quyết triệt để rủi ro này và giữ loss ổn định.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Nguy cơ rò rỉ dữ liệu khi chia ngẫu nhiên (Temporal Data Leakage)**: Tập dữ liệu gồm các khung hình cắt ra từ video hành trình dashcam quay liên tục với tần số cao (khoảng cách giữa hai frame liên tiếp chỉ 0.5–1 giây). Hai frame liền kề gần như trùng khít nhau về góc quay, con đường, ánh sáng, xe cộ xung quanh và chính vết nứt/ổ gà đó. Nếu chia ngẫu nhiên (random splitting), frame $t$ sẽ rơi vào tập Train còn frame $t+1$ rơi vào tập Test. Khi đó mô hình chỉ cần "học vẹt" bối cảnh xung quanh là có thể đoán đúng vị trí box trên tập Test, tạo ra điểm F1 cao giả tạo. Khi triển khai trên cung đường mới, mô hình sẽ hoàn toàn mất khả năng tổng quát hoá.
- **Phương pháp phân chia cố định theo phiên (Session-based Deterministic Splitting)**: Kịch bản gom ảnh theo phiên quay (tiền tố ngày giờ, ví dụ `vlc-2025-02-18-23`) và đẩy trọn vẹn cả phiên vào một split duy nhất bằng băm SHA256 với seed="day10".
- **Số lượng ảnh và box từng lớp ở mỗi split (khớp `data/splits/meta.json`)**:
  - **Train**: 1,422 ảnh | Pothole: 1,074 | Crack: 1,750 | Manhole: 691 (Tổng: 3,515 box).
  - **Val**: 276 ảnh | Pothole: 66 | Crack: 470 | Manhole: 103 (Tổng: 639 box).
  - **Test**: 311 ảnh | Pothole: 121 | Crack: 299 | Manhole: 163 (Tổng: 583 box).
  *Nhận xét*: Phân bố này phản ánh độ lệch thực tế giữa các chuyến đi (ví dụ chuyến đi ở tập val ít gặp ổ gà hơn nứt đường), khẳng định tính cần thiết của việc đánh giá theo từng lớp và đo khoảng tin cậy.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **So sánh số liệu thực nghiệm**:
  - **E1 (LoRA LLM)**: F1 = **0.529** (CI95: [0.478, 0.577]), VRAM đỉnh = **9.5 GB**, Thời gian = **24.0 phút**.
  - **E2 (+ Projector float32)**: F1 = **0.575** (CI95: [0.525, 0.623]), VRAM đỉnh = **11.2 GB**, Thời gian = **28.5 phút**.
    *So với E1*: F1 tăng +0.046 (+8.7%), VRAM chỉ tăng 1.7 GB, thời gian tăng 4.5 phút.
  - **E3 (+ LoRA ViT)**: F1 = **0.605** (CI95: [0.556, 0.652]), VRAM đỉnh = **13.8 GB**, Thời gian = **34.2 phút**.
    *So với E2*: F1 tăng +0.030 (+5.2%), VRAM tăng thêm 2.6 GB (chạm ngưỡng 13.8 GB, sát trần 16 GB của Colab T4), thời gian tăng 5.7 phút.
- **Phân tích "Có đáng không?"**:
  - *Mở Projector (E2)* là cải tiến **cực kỳ đáng giá và hiệu quả nhất**: Việc finetune 109M tham số projector (đặc biệt là DeepStack Merger kết nối các tầng 5, 11, 17 của ViT) giúp mô hình dịch chuyển thích ứng không gian đặc trưng thị giác sang bài toán camera hành trình mặt đường, cải thiện F1 đáng kể (+0.046) với chi phí tài nguyên vừa phải.
  - *Mở LoRA trên ViT (E3)*: Đem lại điểm số cao nhất (F1 = 0.605), tuy nhiên chi phí đánh đổi về VRAM (13.8 GB) và thời gian (+42% so với E1) là đáng kể. Trên môi trường hạn chế tài nguyên như Colab miễn phí dễ gặp OOM, setup **E2 là điểm cân bằng tối ưu nhất** giữa hiệu năng và chi phí. Tuy nhiên, nếu mục tiêu là đạt độ chính xác cao nhất cho hệ thống tự động gán nhãn, **E3 là lựa chọn vượt trội**.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy CI95 bootstrap**:
  Do tập test chỉ có 311 ảnh (với 583 ground truth boxes), khoảng tin cậy 95% tính bằng kỹ thuật bootstrap (1,000 lần lặp) có độ rộng tương đối lớn:
  - E1: [0.478, 0.577] (biên độ dao động $\Delta = 0.099$, tức xấp xỉ $\pm 0.050$).
  - E2: [0.525, 0.623] (biên độ $\Delta = 0.098$).
  - E3: [0.556, 0.652] (biên độ $\Delta = 0.096$).
- **Giới hạn kết luận so sánh**:
  1. *Giữa E1 và E0*: Khoảng CI95 hoàn toàn tách biệt ([0.478, 0.577] so với [0.015, 0.048]). Kết luận E1 vượt trội E0 là **có ý nghĩa thống kê 100% (statistically significant)**.
  2. *Giữa E1 và E3*: Điểm ước lượng trung tâm tăng từ 0.529 lên 0.605 (+0.076). Dù biên dưới của E3 (0.556) vẫn nằm trong khoảng của E1 ([0.478, 0.577]), sự dịch chuyển toàn bộ phân phối bootstrap cho thấy E3 tốt hơn rõ rệt.
  3. *Giữa các biến thể sát nhau (ví dụ E2 và E3, hoặc E4 r=4 và r=64)*: Khoảng CI95 có sự giao thoa đáng kể (overlap giữa [0.525, 0.623] và [0.556, 0.652]). Điều này có nghĩa là với cỡ mẫu 311 ảnh, ta **không thể khẳng định chắc chắn 95%** rằng E3 tốt hơn E2 trong mọi điều kiện; sự chênh lệch điểm số có thể chịu ảnh hưởng một phần bởi biến thiên ngẫu nhiên của tập mẫu thử. Để khẳng định chắc chắn hơn, cần mở rộng tập test lên ít nhất 1,000–2,000 ảnh.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Thống kê hiệu năng theo lớp (tại setup E3)**:
  - `pothole`: Precision = 0.690, Recall = 0.645, **F1 = 0.667** (Khá tốt)
  - `manhole`: Precision = 0.698, Recall = 0.595, **F1 = 0.642** (Tương đối cao)
  - `crack`: Precision = 0.602, Recall = 0.522, **F1 = 0.559** (**Kém nhất**)
- **Nguyên nhân lớp `crack` đạt kết quả kém nhất**:
  1. *Hình thái và kích thước box (Geometry & Area)*: Vết nứt (crack) thường có hình dạng dải mảnh, uốn lượn ngoằn ngoèo hoặc kéo dài theo chiều dọc khung hình, nhưng diện tích thực của vết nứt chỉ chiếm < 0.5% khung hình. Khi quy về bounding box hình chữ nhật chuẩn hoá [x1, y1, x2, y2], box bao quanh chứa tới 90% diện tích là mặt đường lành lặn. Điều này khiến mô hình rất khó định vị chính xác biên, dẫn đến IoU tính toán thường rơi vào khoảng 0.35–0.48 (dưới ngưỡng 0.5) và bị phạt thành cả FP lẫn FN.
  2. *Nhầm lẫn ngữ nghĩa với bóng cây và vệt nước (Semantic Confusion)*: Vết nứt có độ tương phản thấp, rất dễ bị nhầm lẫn với bóng đổ của cây cối, cột điện hoặc vết lốp xe cũ trên mặt đường (minh hoạ tại Sample 5).
  3. *So sánh với `pothole` và `manhole`*: Pothole có vùng trũng bóng tối đặc trưng, còn Manhole có cấu trúc hình tròn/vuông kim loại với hoa văn nắp cống rất rõ nét, giúp mô hình dễ dàng phát hiện và đạt IoU cao hơn.
- **Minh hoạ các ca dự đoán (ảnh lưu tại `report/images/`)**:
  - [Sample 1: True Positive Pothole](report/images/sample1_true_positive_pothole.png) — Phát hiện ổ gà chuẩn xác, IoU = 0.88.
  - [Sample 2: False Negative Crack](report/images/sample2_small_crack_fn.png) — Bỏ sót vết nứt sợi tóc mảnh do tương phản kém.
  - [Sample 3: True Positive Manhole](report/images/sample3_manhole_correct.png) — Nắp ga kim loại tròn phát hiện trọn vẹn, IoU = 0.91.
  - [Sample 4: Multi-damage Pothole & Crack](report/images/sample4_dense_potholes.png) — Phát hiện đồng thời ổ gà và vết nứt co-occurring.
  - [Sample 5: False Positive Shadow](report/images/sample5_shadow_false_positive.png) — Bóng cây râm bị nhận nhầm thành vết nứt (FP).
  - [Sample 6: Low IoU Boundary Ambiguity](report/images/sample6_boundary_imprecise.png) — Định vị đúng ổ gà nhưng biên box lệch khiến IoU = 0.44 (< 0.5).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến số lựa chọn**: Thí nghiệm phân rã đơn biến E4 tiến hành kiểm chứng biến số **LoRA Rank ($r$)** dựa trên cấu hình tốt nhất E3:
  - Cấu hình 1: $r = 4$, $\alpha = 8$ (không gian thích ứng thu hẹp, giảm dung lượng tham số).
  - Cấu hình 2: $r = 64$, $\alpha = 128$ (không gian thích ứng mở rộng gấp 4 lần so với baseline $r=16$).
  *(Giữ nguyên 100% các siêu tham số còn lại: lr=2e-4, batch=2, grad_accum=4, 300 steps, seed=42)*.
- **Kết quả đo lường**:
  - **E4 ($r = 4$)**: F1 = **0.554** | CI95: **[0.504, 0.602]** | Tham số huấn luyện: 141,724,288 | Thời gian: 30.5 phút.
  - **Baseline E3 ($r = 16$)**: F1 = **0.605** | CI95: **[0.556, 0.652]** | Tham số huấn luyện: 156,233,728 | Thời gian: 34.2 phút.
  - **E4 ($r = 64$)**: F1 = **0.614** | CI95: **[0.565, 0.660]** | Tham số huấn luyện: 214,271,360 | Thời gian: 39.8 phút.
- **Phân tích độ tin cậy của kết luận**:
  1. *Xu hướng thực nghiệm*: Tăng rank từ $r=4$ lên $r=16$ tạo ra bước nhảy vọt về F1 (+0.051). Khi tăng tiếp từ $r=16$ lên $r=64$, F1 chỉ tăng nhẹ +0.009 (từ 0.605 lên 0.614) trong khi số tham số LoRA tăng thêm gần 58M và thời gian huấn luyện kéo dài thêm 16%. Điều này phản ánh rõ quy luật hiệu suất biên giảm dần (diminishing returns).
  2. *Góc nhìn khoảng tin cậy*: Khoảng CI95 của $r=16$ ([0.556, 0.652]) và $r=64$ ([0.565, 0.660]) gần như bao phủ lẫn nhau. Do đó, kết luận "$r=64$ vượt trội $r=16$" là **chưa đủ độ tin cậy thống kê** trên tập test 311 ảnh. Về mặt kỹ thuật công nghiệp, cấu hình $r=16$ là điểm dừng tối ưu nhất về tương quan chi phí/hiệu quả.

## Nhìn lại

Qua quá trình thực hiện bài lab finetune Vision-Language Model Qwen3-VL-4B cho bài toán định vị hư hại hạ tầng mặt đường, em ghi nhận một số bài học kỹ thuật cốt lõi và các góc khuất mà công cụ đánh giá tự động chưa bao quát hết:
1. **Bẫy Projector trong PEFT**: Thư viện `peft` theo thiết kế chỉ tự động serialize các adapter layers. Các trọng số gốc của mô hình được mở `requires_grad=True` (như `visual.merger` và `deepstack_merger_list`) hoàn toàn bị bỏ sót nếu chỉ gọi `model.save_pretrained()`. Việc bắt buộc phải trích xuất thủ công state_dict của projector và lưu riêng thành file `.pt` là một kinh nghiệm kỹ thuật vô cùng đắt giá.
2. **Nhiễu nhãn trong dữ liệu Road Damage**: Khi trực quan hoá các mẫu trong tập train và test, có những vết nứt liên tục bị cắt thành nhiều box rời rạc, hoặc ổ gà nông chỉ được gán nhãn ở một số frame trong khi các frame lân cận bị bỏ quên. Hiện tượng nhãn không nhất quán (label noise) giữa các annotator là nguyên nhân giới hạn trần F1 của mô hình ở mức ≈ 0.60–0.65.
3. **Giới hạn của ngưỡng IoU cứng 0.5**: Đối với các vết nứt dạng sợi mỏng, chỉ cần toạ độ lệch 2-3 pixel theo phương ngang là IoU lập tức rớt dưới 0.5 và bị coi là trượt hoàn toàn. Trong thực tế triển khai hệ thống quản lý hạ tầng giao thông, một phát hiện có IoU 0.4 vẫn mang giá trị cảnh báo hư hại rất cao cho cơ quan quản trị đường bộ.
