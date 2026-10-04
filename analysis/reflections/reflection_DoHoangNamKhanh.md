# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đỗ Hoàng Nam Khánh  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Với `threshold=0.5`, nhóm câu theo cosine similarity (`all-MiniLM-L6-v2`). Khi chạy trên corpus 25 file, semantic tạo ít chunk hơn basic ở nơi các câu cùng chủ đề liền kề. Khi chưa có model, tôi phải fallback lexical (scale threshold ×0.4) — minh chứng rõ rằng **embedding thật quan trọng**: lexical similarity bỏ sót liên kết ngữ nghĩa giữa câu "12 ngày" và "thâm niên". |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent ~2048 ký tự gộp nhiều đoạn, child ~256 ký tự để retrieve chính xác rồi **trả parent cho LLM**. Tôi kiểm tra được `parent_id` liên kết hợp lệ và children nhỏ hơn parents. Khi pipeline thực hiện đúng "retrieve child → return parent", Context Recall tăng từ 0.80 (naive) lên **0.925** — đây là cải thiện lớn nhất của cả bài. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Dùng `re.split(r'(^#{1,3}\s+.+$)', ..., MULTILINE)` để cắt theo header. Giữ được tiêu đề + bảng nguyên vẹn — quan trọng với `bang_luong_2024.md` và `mua_sam.md` vì bảng bị cắt sẽ mất thông tin thẩm quyền phê duyệt. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF (`Σ 1/(k+rank+1)`) kết hợp danh sách lexical (BM25, khớp từ khóa/số chính xác) và semantic (dense). Nhờ RRF, chunk xuất hiện ở **cả hai** danh sách được đẩy lên top, giải quyết điểm yếu "BM25 bỏ lỡ đồng nghĩa, dense bỏ lỡ con số". |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | `underthesea.word_tokenize(format="text")` nối từ ghép bằng `_` ("nghỉ_phép"); phải `replace("_", " ")` để BM25 `split(" ")` khớp query. Nếu không, `"nghỉ_phép"` (1 token) không khớp `"nghỉ phép"` (2 token). |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Rerank top-20 → top-3 bằng `sentence_transformers.CrossEncoder` (`bge-reranker-v2-m3`). Sau khi cài model thật + trả parent context, Context Precision đạt **0.90** (so với 0.858 của naive) và chunk đúng thường lên top-1. Khi thiếu model, lexical fallback khiến precision giảm rõ — đúng như kỳ vọng: **reranking là mắt xích quyết định precision**. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Wrap `try/except`; chạy **RAGAS thật** với LLM OpenCode Go (`deepseek-v4.1-flash`) làm judge và embedding local `bge-m3`. Kết quả production: Faithfulness 0.745, Answer Relevancy 0.842, Context Precision 0.900, Context Recall 0.925 (`engine: "ragas"`). Khi thiếu provider thì fallback sang metric lexical proxy (`engine: "heuristic"`). |
| Failure analysis / Diagnostic Tree | M4 | `failure_analysis()` | Map metric yếu nhất → nguyên nhân → cách sửa. Bottom-5 cho thấy đa số lỗi thuộc **retrieval/rerank** (context_precision thấp) chứ không phải generation. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Gộp 4 kỹ thuật vào **1 lời gọi LLM/chunk** (chạy thật trên 109 chunk) để tiết kiệm chi phí; mỗi chunk được chèn 1 câu context mô tả nguồn trước khi embed. Khi offline thì fallback chèn `"Trích từ <source>."`. Việc prepend giúp chunk tự mô tả nguồn, giảm retrieval failure kiểu Anthropic benchmark. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `ModuleNotFoundError: No module named 'pypdf'` khiến `test_compare_all_strategies` fail vì `load_documents()` cố đọc 3 file PDF (`BCTC.pdf`, `Nghi_dinh...pdf`, `so_tay_an_toan.pdf`).
  - `AssertionError: assert 8 <= (5 + 2)` trong `test_semantic_groups_by_topic` — semantic fallback tách quá nhiều chunk.
  - `ModuleNotFoundError: No module named 'sentence_transformers'` và `'ragas'` khi chạy pipeline.
  - `AttributeError: module 'numpy' has no attribute 'long'` khi import `sklearn`/`sentence_transformers` — do `ragas<0.2` kéo `numpy` xuống `1.26.4` trong khi `scipy 1.18` lại đòi `numpy>=2.0`.
- **Nguyên nhân gốc rễ & Cách debug:**
  - **pypdf:** liệt kê `data/*.pdf` mới nhận ra corpus có PDF dù trước đó chỉ thấy `.md`. Cài `pip install pypdf rank-bm25` → test pass. Bài học: **đọc kỹ data inventory**, đừng đoán theo README.
  - **semantic fallback:** In ra similarity giữa từng cặp câu để thấy Jaccard của câu cùng chủ đề chỉ ~0.2–0.27, thấp hơn nhiều so với cosine của embedding thật. Giải pháp: khi không có embedding, dùng `sim_threshold = threshold * 0.4` (vì lexical similarity có biên độ thấp hơn). Sau đó 8 → 5 chunk, test pass.
  - **thiếu model:** thiết kế mọi điểm phụ thuộc (`pypdf`, `underthesea`, `rank_bm25`, `qdrant`, `sentence_transformers`, `ragas`) theo **try/except + fallback** để pipeline luôn chạy end-to-end, kể cả khi môi trường offline/không có Docker/API key.
  - **numpy/scipy conflict:** đọc kỹ warning của pip (`scipy 1.18 requires numpy>=2.0, but you have numpy 1.26.4`), rồi pin lại `scipy>=1.11,<1.14` để tương thích `numpy 1.26.4` mà RAGAS yêu cầu → `sklearn`/`sentence-transformers` import lại bình thường.
  - **API key lạ bị inject:** pipeline bỗng gọi OpenAI và trả về `NaN` vì môi trường có key không hợp lệ (`oc_sk_...`). Tôi thêm guard trong `config.py` (chỉ nhận key bắt đầu bằng `sk-`) và trong `evaluate_ragas()` (nếu kết quả toàn `NaN` thì fallback heuristic) để pipeline ổn định và nhanh trở lại.
  - **Dùng key OpenCode Go cho LLM:** `oc_sk_...` là key OpenCode Go chứ không phải OpenAI. Tôi tìm endpoint OpenAI-compatible `https://opencode.ai/zen/go/v1`, phát hiện yêu cầu header `x-opencode-session` (nếu thiếu báo `MissingSessionID`), và **không có endpoint `/embeddings`** (trả 404). Giải pháp: dùng OpenCode Go cho chat/judge, còn **embedding chạy local bằng `bge-m3`** rồi truyền vào RAGAS qua `LangchainEmbeddingsWrapper`. Toàn bộ được gói trong `src/llm.py`. Kết quả: RAGAS chạy thật (`engine: "ragas"`).
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Hiểu sâu `reciprocal_rank_fusion` và vì sao dùng `1/(k+rank)` thay vì cộng điểm thô (điểm BM25 và cosine khác thang đo, RRF chỉ dựa thứ hạng nên robust).
  - Cách RAGAS gọi LLM-judge và vì sao cần `OPENAI_API_KEY`; tìm hiểu qua docs RAGAS và tự viết fallback metric để không bị chặn.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý hỏi đáp chính sách nội bộ (HR/IT Helpdesk Bot)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Basic RAG — paragraph chunking + dense-only (embedding nhỏ) + top-3 context, không hybrid, không rerank, không enrichment.
- **Vấn đề / Bottlenecks đang gặp:** Trả lời sai các câu hỏi **version** (chính sách cũ/mới), hỏi **multi-hop/numeric** (thẩm quyền phê duyệt theo số tiền) và câu **negation** ("có được ... không?"). Context precision thấp vì top-3 lẫn chunk chéo chủ đề.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** **Hierarchical** (parent 2048 / child 256) làm mặc định; dùng **structure-aware** riêng cho file có bảng (`mua_sam`, `bang_luong`) để không cắt vỡ bảng.
2. **Search retrieval:** **Hybrid BM25 + Dense + RRF**; bật `underthesea` để segment tiếng Việt đúng, thêm numeric-aware tokenization cho câu có số/đơn vị.
3. **Reranking:** **Có** — `CrossEncoder bge-reranker-v2-m3` (top-20 → top-3); fallback flashrank nếu cần latency thấp (<5ms).
4. **Evaluation:** **RAGAS 4 metrics** trên test set 20 câu phân loại theo 6 dạng (lookup/version/negation/multi-hop/numeric/ambiguous); thêm metric tùy chỉnh **exact-number match** và **version-awareness**.
5. **Enrichment:** **Contextual prepend** (giảm retrieval failure) + **auto metadata** (`category`, `version`) để filter chéo chủ đề; giữ combined 1-call/chunk để tiết kiệm chi phí.

#### 3. Timeline triển khai
- **Tuần 1:** Chuẩn hoá corpus + hierarchical/structure-aware chunking; cài embedding `bge-m3` thật; index Qdrant.
- **Tuần 2:** Hybrid search + RRF, đánh giá offline; bật reranker cross-encoder; chạy RAGAS baseline vs production.
- **Tuần 3:** Enrichment combined + metadata filter; xử lý riêng nhóm câu version/negation/multi-hop; viết failure analysis và tinh chỉnh prompt.
- **Tuần 4:** Đo latency từng bước, tối ưu cache/embedding; đóng gói demo và viết tài liệu.
