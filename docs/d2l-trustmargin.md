# Doc-to-LoRA + TrustMargin

## Phạm vi tích hợp

Luồng `/ask` chạy Doc-to-LoRA trực tiếp trong tiến trình FastAPI. Model đọc
`base_model_name_or_path` từ checkpoint và dùng tokenizer tương ứng của repo
`doc-to-lora` đã tích hợp tại `vendor/doc-to-lora`. Vì vậy model RAG và model nền D2L luôn cùng danh tính; không cấu
hình một tên model độc lập ở repo RAG.

Retriever BGE-M3 / Qdrant / reranker vẫn giữ nguyên. Với mỗi câu hỏi, lấy top-8
đoạn, ghép theo thứ tự truy xuất, rồi giới hạn bằng tokenizer của cả model trả lời
và context encoder. Cùng chuỗi bằng chứng P được dùng cho:

1. **RAG:** model nền đã reset adapter, prompt chứa câu hỏi q và P.
2. **D2L:** `internalize(P)` sinh LoRA bằng hypernetwork; model có adapter trả lời
   prompt chỉ chứa q. Không có bước huấn luyện gradient trong lượt hỏi.
3. **TrustMargin:** chấm hai câu trả lời có sẵn rồi chọn một câu.

Đây là D2L trên các đoạn đã truy xuất, **không phải nạp toàn bộ PDF khi upload**.
Adapter được tạo theo lượt hỏi và xóa sau lượt, kể cả khi lỗi. Chưa lưu adapter
theo tài liệu; cách này tránh trộn trạng thái giữa các phiên nhưng tốn thêm thời gian.

## Ánh xạ từ paper

Nguồn: *TrustMargin: Training-Free Arbitration between Parametric Memory and
Retrieved Evidence in Large Language Models*, arXiv:2606.08397v1, mục 3 và phụ lục D.

Paper gốc sinh Direct/RAG từ cùng một model đóng băng. Ở đây thay Direct bằng
câu trả lời D2L; RAG vẫn sinh từ model nền không gắn adapter. **Đây là mở rộng của
TrustMargin, không phải tái lập nguyên trạng thiết lập thực nghiệm trong paper.**

Để prior phản ánh bộ nhớ đã nạp tài liệu, cả sáu likelihood được chấm với **cùng
trạng thái D2L đã đóng băng** sau khi sinh ứng viên D2L. Không đổi adapter giữa
các ứng viên hoặc giữa ba view. Như vậy prior ở đây đã được điều kiện hóa gián tiếp
bởi P qua adapter; không còn là prior trước khi tiếp xúc tài liệu như paper gốc.
Đây là giả thuyết triển khai cần kiểm chứng trên dữ liệu tiếng Việt.

Gọi yD là câu trả lời D2L, yR là câu trả lời RAG. Với mỗi câu y:

```text
lD(y) = mean token log p(y | q; fixed D2L adapter)
lR(y) = mean token log p(y | q, P; fixed D2L adapter)
lC(y) = mean token log p(y | P; fixed D2L adapter)

Mprior = lD(yR) - lD(yD)
Mbind  = [lR(yR) - lC(yR)] - [lR(yD) - lC(yD)]
M      = Mprior + lambda_bind * Mbind

Chọn RAG nếu M > tau; ngược lại chọn D2L (kể cả bằng ngưỡng).
```

Teacher forcing chỉ tính log-likelihood trung bình trên token câu trả lời, dịch
logits một vị trí; không tính token prompt, padding hoặc token điều khiển. Không
dùng điểm reranker, điểm tự đánh giá do LLM viết ra hoặc xác suất từ decoding làm
thay thế. Không sinh câu trả lời thứ ba. Prompt context-only bỏ câu hỏi hoàn toàn.

Mặc định `lambda_bind=0.5`, `tau=-1.5` lấy từ paper. Đây là điểm khởi đầu, **chưa
được hiệu chỉnh cho D2L/Gemma và dữ liệu tiếng Việt**. M không phải xác suất đúng.
Phải so sánh `auto`, `rag`, `d2l` trên một tập validation riêng, theo dõi chất lượng,
tỷ lệ chọn nhánh và độ trễ; chọn ngưỡng trên validation rồi đánh giá trên tập test
không dùng để chọn ngưỡng. Hai ứng viên cùng sai thì bộ chọn không sửa được.

## Chạy hệ thống trong một môi trường

Phần suy luận upstream đã được đưa vào `vendor/doc-to-lora`, giữ namespace
`ctx_to_lora`, giấy phép MIT và mã commit nguồn. Runtime được cài như một Python
package nội bộ qua `uv sync`. Không cần repo `D:/doc-to-lora` tồn tại khi chạy,
không có tiến trình model riêng, không gọi HTTP cho generation.

Bộ thư viện chung dùng Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0,
Datasets 3.6.0, FlagEmbedding 1.3.5 và Sentence Transformers 3.4.1. `uv.lock`
đã được cập nhật, giữ các phiên bản khác nếu có thể. Không cài lại các pin
Transformers/PEFT cũ của RAG.

### 1. Cài thư viện và checkpoint

Từ thư mục gốc `vietnamese-rag-system`:

```bash
uv sync --locked --dev
uv run huggingface-cli login
uv run huggingface-cli download SakanaAI/doc-to-lora --local-dir trained_d2l --include '*/'
```

Nếu checkpoint dùng model gated, tài khoản Hugging Face phải được cấp quyền model
nền tương ứng. Chỉ load checkpoint đáng tin cậy: upstream dùng `torch.load` với
`weights_only=False` vì checkpoint chứa các lớp cấu hình Python.

Model production vẫn cần PyTorch có CUDA và GPU đủ bộ nhớ. Cài bản CUDA thích hợp
với phiên bản Torch trong lockfile vào chính `.venv` của RAG theo hướng dẫn PyTorch;
kiểm tra trước khi khởi động:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Giá trị CUDA phải là `True`. Không cần cài Flash Attention: bản tích hợp dùng SDPA
của PyTorch cho Perceiver không đóng gói chuỗi. Các tên và kích thước tensor trọng
số được giữ nguyên; cross-attention dùng context làm key/value như upstream.
Nếu checkpoint bật `quantize_ctx_encoder`, cần cài thêm `bitsandbytes>=0.46.1`
vào cùng môi trường, phù hợp nền tảng CUDA. Mặc định upstream không bật tùy chọn này.

### 2. Cấu hình

Bổ sung vào `.env` (tham khảo `.env.example`):

```env
D2L_CHECKPOINT_PATH=trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin
D2L_MAX_INPUT_TOKENS=4096
D2L_MAX_CONTEXT_TOKENS=2048
LLM_MAX_NEW_TOKENS=256
TRUSTMARGIN_LAMBDA_BIND=0.5
TRUSTMARGIN_TAU=-1.5
```

Có thể đặt đường dẫn tuyệt đối đến checkpoint sẵn có trong repo `doc-to-lora`.
Đường dẫn tương đối được tính từ thư mục gốc RAG. Không cần biến cấu hình URL model.
Tên model nền luôn đọc từ checkpoint, không có model generation độc lập cho RAG.
Chat template được đọc từ package, không phụ thuộc thư mục làm việc của server.

### 3. Khởi động

Khởi động Qdrant, Redis như trước, rồi chạy API và giao diện:

```bash
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
uv run streamlit run src/ui/streamlit_app.py
```

FastAPI nạp checkpoint một lần khi khởi động; thiếu checkpoint/CUDA sẽ báo lỗi
khởi động rõ ràng. Toàn bộ generation, internalization và scoring chạy trong một
thread của **cùng tiến trình**, giữ vòng lặp API thông suốt để gửi heartbeat.
Không dùng subprocess. Khi tắt API, hệ thống chờ tính toán đang chạy kết thúc rồi
nhả model; mỗi lượt có `finally: reset()` để xóa adapter.

Chỉ chạy một process API cho một bản model. Nếu dùng nhiều process, mỗi process
sẽ nạp một bản model riêng và tăng lượng VRAM tương ứng. Embedding, reranker và D2L
cùng dùng tài nguyên của tiến trình này.

Giao diện có ba chế độ: Tự động (TrustMargin), RAG, Doc-to-LoRA. API phía giao diện:

```text
GET /ask?query=...&session_id=...&mode=auto
```

Sự kiện NDJSON gồm `status`, `sources`, `routing`, `content`, hoặc `error`. Trong
`auto`, phải sinh xong hai ứng viên và chấm sáu likelihood trước khi gửi nội dung;
trong lúc chờ vẫn gửi heartbeat. Chế độ cưỡng bức chỉ sinh một nhánh.

`routing` chứa nhánh được chọn, model, Mprior, Mbind, M, lambda, tau, sáu likelihood
và cờ cắt ngắn context. Nguồn trả về là những đoạn đã dùng sau giới hạn token;
chúng là bằng chứng đầu vào, không chứng minh từng phát biểu đều được hỗ trợ.
Cache bao gồm câu hỏi nguyên bản, tài liệu, chế độ, tham số và định danh instance
model. Khởi động lại API hoặc đổi tài liệu sẽ không lấy lại quyết định cũ.

## Kiểm thử và giới hạn

```bash
uv run pytest tests/ -q
```

Kiểm thử gồm công thức/ngưỡng, teacher forcing, adapter/reset/đồng thời, vòng đời
model trong API, cache và hủy request. Ngoài model giả, có kiểm thử chạy **mã
hypernetwork upstream thật** với Llama nhỏ khởi tạo ngẫu nhiên trên CPU: sinh LoRA,
sinh hai ứng viên, chấm sáu likelihood và reset; không tải trọng số từ Internet.
Attention SDPA được đối chiếu với phép attention tính tường minh và kiểm tra mask.
Những test này không thay thế đánh giá checkpoint pretrained và chất lượng QA.

Máy kiểm tra có RTX 3050 4 GB, môi trường đang dùng Torch CPU và chưa có checkpoint
D2L. Chưa xác nhận model pretrained chạy trên GPU hoặc vừa bộ nhớ máy này. Tích hợp
không triển khai CPU/offload cho checkpoint production. Ngắt request không dừng
kernel đang chạy; model vẫn bị khóa đến khi hoàn tất. Chưa có hàng đợi bền vững
hoặc giới hạn số request chờ.
