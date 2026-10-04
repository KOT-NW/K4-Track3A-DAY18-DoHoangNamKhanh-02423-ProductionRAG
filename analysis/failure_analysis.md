# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Đỗ Hoàng Nam Khánh  
**Khóa:** K4 - Track 3A  

> Lần chạy này dùng **LLM thật (OpenCode Go — `deepseek-v4.1-flash`)** làm judge cho
> RAGAS và làm generator, kết hợp embedding `BAAI/bge-m3`, reranker
> `BAAI/bge-reranker-v2-m3`, `underthesea` và Qdrant in-memory.
> `reports/ragas_report.json` có `"engine": "ragas"` (điểm RAGAS chính thức).

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9000 | 0.7450 | −0.1550 |
| Answer Relevancy | 0.8147 | 0.8422 | +0.0275 |
| Context Precision | 0.8583 | 0.9000 | +0.0417 |
| Context Recall | 0.8000 | 0.9250 | +0.1250 |

**Nhận xét:**
- Production thắng rõ ở **Context Recall (+0.125)** và **Context Precision (+0.042)** —
  nhờ hierarchical chunking + **retrieve child → return parent**, hybrid BM25+dense+RRF
  và cross-encoder rerank. Đây là mục tiêu chính của pipeline nâng cao.
- **Answer Relevancy** nhích nhẹ (+0.028).
- **Faithfulness giảm (−0.155)**: context (parent) lớn hơn giúp lấy đủ evidence, nhưng
  generator đôi khi **suy luận thêm** (ví dụ tự tính `85% × 20.000.000 = 17.000.000`)
  trong khi parent context không chứa con số 20.000.000 → RAGAS đánh giá là không được
  context hỗ trợ. Đây là đánh đổi recall↑ / faithfulness↓ điển hình.
- Production đạt **cả 4 metric ≥ 0.70** (rubric #7: mức cao nhất).

---

## Bottom-5 Failures

### #1 — Senior 9 năm thâm niên: phép năm + khoảng lương (multi-hop)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + (9÷3) = **18 ngày**; lương Senior P3–P4: **20–35 triệu**.
- **Got:** "Theo chính sách 2024: 9 năm thâm niên được **18 ngày phép năm** (15 + 3). Về mức lương, **không tìm thấy**."
- **Worst metric:** answer_relevancy = **0.00** (avg 0.29)
- **Error Tree:** Output thiếu → Context đúng? **Nửa đầu đúng, nửa sau thiếu** (context chỉ có `nghi_phep_nam_v2024` + `v2023`, **không có chunk bảng lương**) → Query OK? **Multi-hop** → **Root cause: retrieval không phục vụ ý thứ hai (lương)**.
- **Root cause:** Một truy vấn trộn hai chủ đề; hybrid search dồn về chủ đề "nghỉ phép" và bỏ sót bảng lương.
- **Suggested fix:** **Query decomposition** (tách "phép năm" / "khung lương Senior") rồi hợp nhất kết quả; metadata filter theo `category`.

### #2 — Tạm ứng 15 triệu, trả sau 20 ngày (numeric)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; phí 2%/tháng trên 15.000.000 = **300.000 VNĐ/tháng** (pro-rata ~50.000 VNĐ cho 5 ngày).
- **Got:** "Phạt **2%/tháng** trên 15.000.000, tức **300.000 VNĐ/tháng**." (đúng phần chính, thiếu pro-rata).
- **Worst metric:** faithfulness = 0.33 (avg 0.68)
- **Error Tree:** Output đúng một phần → Context đúng? **Có** (`tam_ung.md`) → Query OK? **OK** → **Root cause: generator bỏ qua bước pro-rata** khiến câu trả lời không khớp hoàn toàn với ground truth.
- **Suggested fix:** Prompt yêu cầu tính toán từng bước khi câu hỏi có số; few-shot ví dụ pro-rata.

### #3 — Lương thử việc Junior mức cao nhất (numeric + suy luận)
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior max 20.000.000 → 85% = **17.000.000 VNĐ/tháng**.
- **Got:** "Lương thử việc = 85% × 20.000.000 = **17.000.000 VNĐ/tháng**." (kết quả đúng).
- **Worst metric:** faithfulness = **0.00** (avg 0.69)
- **Error Tree:** Output đúng kết quả → Context đúng? **Thiếu con số 20.000.000** trong parent context → **Root cause: answer suy luận vượt quá context** → RAGAS coi là hallucination dù đáp án đúng.
- **Root cause:** Bảng lương bị chunk theo hierarchical nên hàng "Junior" và mốc 20M tách khỏi parent context.
- **Suggested fix:** **Structure-aware chunking cho bảng lương** để giữ nguyên khung lương; hoặc truyền thêm chunk `bang_luong_2024.md` vào context.

### #4 — Thâm niên bao nhiêu năm thì cộng ngày phép (version)
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** v2024: từ **3 năm**, mỗi 3 năm +1 ngày; v2023 yêu cầu 5 năm.
- **Got:** "Theo chính sách 2024: từ **3 năm trở lên**, +1 ngày mỗi 3 năm." (đúng).
- **Worst metric:** context_precision = 0.50 (avg 0.74)
- **Error Tree:** Output đúng → Context đúng? **Đúng nhưng lẫn v2023** (precision thấp) → Query OK? **OK** → **Root cause: version cũ lọt vào top context**.
- **Suggested fix:** Metadata `version`/`effective_date` + filter bản `current`; auto-metadata (M5) đã trích nhưng cần dùng để filter.

### #5 — Có cần kích hoạt MFA không? (negation + version)
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** v2.0 bắt buộc MFA; v1.0 không yêu cầu.
- **Got:** "Có. Tất cả nhân viên **bắt buộc** kích hoạt MFA cho email, VPN và hệ thống nội bộ." (đúng).
- **Worst metric:** context_recall = 0.50 (avg 0.80)
- **Error Tree:** Output đúng → Context đúng? **Đúng nhưng thiếu chunk v1.0 để so sánh** và có `WFH` lẫn vào → **Root cause: recall thiếu nhánh so sánh version**.
- **Suggested fix:** Với câu hỏi yes/no + version, retrieve cả bản cũ và mới; metadata filter chủ đề `it`/`password`.

---

## Case Study (cho presentation)

**Question chọn phân tích:** #1 — Senior 9 năm thâm niên (multi-hop: phép + lương).

**Error Tree walkthrough:**
1. Output đúng? → **Một nửa** — đúng 18 ngày phép, thiếu khoảng lương.
2. Context đúng? → **Thiếu** chunk bảng lương; chỉ có 2 chunk chính sách nghỉ phép.
3. Query rewrite OK? → Không; câu hỏi gồm **2 ý** nhưng giữ nguyên 1 truy vấn.
4. Fix ở bước: **Retrieval** — query decomposition + metadata category (`nghi_phep` / `bang_luong`).

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Query decomposition** cho câu multi-hop (Senior phép+lương; laptop phê duyệt+CNTT).
- **Structure-aware chunking cho bảng** (`bang_luong_2024`, `mua_sam`) để giữ hàng/cột.
- **Metadata filter `version`/`category`** trước RRF (dùng auto-metadata M5) để loại v2023 và nhiễu chéo chủ đề.
- **Prompt chống suy luận vượt context** (few-shot numeric, yêu cầu trích dẫn) để tăng Faithfulness từ 0.745.
