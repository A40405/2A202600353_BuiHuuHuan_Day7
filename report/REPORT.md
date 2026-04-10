# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Bùi Hữu Huấn  
**Nhóm:** C401-E2  
**Ngày:** 10/04/2026  

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

High cosine similarity biểu thị hai vector có hướng gần giống nhau → nghĩa là hai câu có ngữ nghĩa tương tự.

- Ví dụ HIGH:
  - "Con chó đang ngủ trên thảm"
  - "Một con cún đang ngủ trên tấm thảm"

- Ví dụ LOW:
  - "Con chó đang ngủ"
  - "Lãi suất ngân hàng tăng"

Cosine similarity tốt hơn Euclidean vì không phụ thuộc độ dài vector.

---

### Chunking Math (Ex 1.2)

- chunk_size = 500, overlap = 50
- stride = 450
- (10000 - 50)/450 ≈ 22.11 → 23 chunks

Nếu overlap = 100 → ~25 chunks

Overlap giúp giữ context giữa các chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain
Y tế (Vinmec)

### Lý do chọn
- Dữ liệu có cấu trúc rõ
- Nhiều nội dung chuyên môn
- Phù hợp test RAG

### Metadata Schema
- source: tên file
- category: loại bệnh

---

## 3. Chunking Strategy (15 điểm)

### Strategy của tôi

**Loại:** Recursive Character Splitting

### Mô tả
- Chia theo thứ tự: đoạn → dòng → câu → ký tự
- Dùng recursion
- Đảm bảo không vượt chunk_size

### Ưu điểm
- Giữ context tốt hơn fixed
- Linh hoạt

### Nhược điểm
- Có thể cắt giữa câu
- Chunk không đồng đều

---

### Code snippet (nếu custom)

```python
# Recursive Character Splitting (simplified)
class RecursiveChunker:
    DEFAULT_SEPARATORS = ["

", "
", ". ", " ", ""]

    def __init__(self, chunk_size=500, separators=None):
        self.chunk_size = chunk_size
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, seps: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()]
        if not seps:
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = seps[0]
        if sep == "":
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        chunks, buf = [], ""
        for p in parts:
            cand = buf + (sep if buf else "") + p
            if len(cand) <= self.chunk_size:
                buf = cand
            else:
                if buf:
                    chunks.extend(self._split(buf, seps[1:]))
                buf = p
        if buf:
            chunks.extend(self._split(buf, seps[1:]))
        return [c.strip() for c in chunks if c.strip()]
```

---

### Strategy của tôi vs Baseline

```python

```

### Bảng so sánh mẫu (trích từ chạy thực tế)


| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `aids-4699.md` | RecursiveChunker (Best Baseline) | 21 | 329.76 | Tốt nhưng đôi chỗ bị chia quá vụn. |
| `aids-4699.md` | Document-structure | 17 | 404.0 | Cấu trúc bệnh lý nằm trọn trong một chunk. |
| `am-anh-so-hai-4678.md` | RecursiveChunker (Best Baseline) | 24 | 354.08 | Tốt, giữ được ngữ nghĩa cấp độ đoạn văn. |
| `am-anh-so-hai-4678.md` | Document-structure | 17 | 496.71 | Gom trọn phương pháp điều trị không bị cắt nhỏ quá nhiều. |
| `ap-xe-nao-3205.md` | RecursiveChunker (Best Baseline) | 35 | 292.29 | Khá rời rạc, quá nhiều chunk nhỏ làm loãng context. |
| `ap-xe-nao-3205.md` | Document-structure | 20 | 508.95 | Dấu hiệu và biến chứng nằm chung bối cảnh. |

---
### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Mạc Phạm Thiên Long | Document-structure Chunking | 8.5 | Nội dung liền mạch, kết quả 4/5 Top-3 | File quá dài có thể sẽ làm chunk quá to |
| Nguyễn Doãn Hiếu | RecursiveChunker | 8.0 | Chia chunk linh hoạt, kích thước vừa phải | Dễ cắt đứt nội dung của danh sách/phác đồ |
| Cao Chí Hải | Semantic Chunking | 6.5 | Q2 precision=1.0; chunk=0.5 tạo 468 chunks coherent, giữ nguyên ý tốt với đoạn dài | Q3 fail hoàn toàn khi search≥0.5; `vi_retrieval_notes` át các file khác do cross-lingual gap | Dùng chunk=0.5 + search=0.4 + top_k=3; thêm metadata filter theo ngôn ngữ |
| Phan Hoài Linh| Semantic Chunking | 7.0 | Giữ nguyên ý tưởng, phù hợp với đoạn dài|  Phụ thuộc vào chất lượng embedding |
| Bùi Hữu Huấn | Recursive Character Splitting | 6.7 / 10 | - Retrieval tốt với 2/3 query đạt tối đa (Amip, Alzheimer)<br>- Chunk giữ ngữ nghĩa khá ổn (avg ~319, không quá ngắn)<br>- Grounding rất tốt (3/3 câu trả lời đúng keyword)<br>- Score distribution có phân tách (spread ~0.2) | - Fail hoàn toàn 1 query (Áp xe gan → 0/2)<br>- Metadata filter chưa hiệu quả (filter không trả kết quả)<br>- Chunk vẫn có khả năng bị cắt giữa câu (min length 25)<br>- Recall chưa ổn định giữa các domain |
| Nguyễn Văn Đạt | Document-structure (custom) + OpenAI embeddings | 10.0 | 5/5 queries top-1 đúng expected_source (hit@5=1.00, MRR=1.00) | Chưa chứng minh hơn baseline (đang hòa); tốn chi phí/độ trễ do gọi API |

**Strategy nào tốt nhất cho domain này? Tại sao?**
Theo kết quả đánh giá (bảng `Kết Quả Của Mạc Phạm Thiên Long và Nguyễn Văn Đạt`), chiến lược **Document-structure Chunking** do Kết Quả Của Mạc Phạm Thiên Long và Nguyễn Văn Đạt lựa chọn hoạt động tốt nhất. Đặc thù tài liệu Vinmec là chia thành các phần rõ rệt (Nguyên nhân, Triệu chứng, Điều trị), chia cắt theo cấu trúc Markdown (`#`) giúp không làm rách mạch phác đồ (medical regimens), giúp RAG luôn có đủ lượng Text để trả lời trọn vẹn câu mà không lo bị cắt xén lưng chừng.

## 4. My Approach (10 điểm)

### Chunking

**`RecursiveChunker.chunk` / `_split`** — approach:
Áp dụng thuật toán chia để trị (divide and conquer) thông qua đệ quy. Văn bản sẽ được tách bằng các ký tự phân cách theo thứ tự ưu tiên giảm dần: từ đoạn văn (`\n\n`), dòng (`\n`), câu (`. `) cho đến khi đạt được kích thước nhỏ hơn `chunk_size`. Điểm then chốt là nếu một phần vẫn quá lớn sau khi dùng hết các separator, dùng `FixedSizeChunker` làm phương án dự phòng cuối cùng.

**`Điểm quan trọng`** là:

    Nếu chunk vẫn quá lớn → tiếp tục chia nhỏ với separator cấp thấp hơn
    Nếu không còn separator phù hợp → fallback sang chia theo fixed-size

    → đảm bảo mọi chunk đều hợp lệ và không vượt quá giới hạn.

### Embedding
** `LocalEmbedder (Sentence-Transformers: all-MiniLM-L6-v2)`** — approach:
    Sử dụng mô hình embedding nhẹ, phổ biến cho semantic search. Vector đầu ra được chuẩn hóa (normalize), do đó phép dot product tương đương cosine similarity.

**`Ưu điểm`**:

    Nhanh, chạy local
    Không phụ thuộc API

**`Nhược điểm`**:

    Chất lượng embedding có thể không tốt bằng các mô hình lớn hơn (như OpenAI)
    Chưa tối ưu hoàn toàn cho domain y tế chuyên sâu

### Vector DB (Qdrant)
**`add_documents`** — approach:
    Mỗi chunk được chuyển thành vector embedding và lưu vào Qdrant dưới dạng PointStruct gồm:

    id: UUID (tránh trùng lặp)
    vector: embedding
    payload: chứa content và metadata

**`Cấu hình collection`**:

    Vector size: 384
    Distance: Cosine
### Retrieval (cosine similarity)
**`search`** — approach:

    Embed câu hỏi
    Truy vấn Qdrant bằng query_points()
    Lấy top-k chunk có score cao nhất
Score phản ánh độ tương đồng ngữ nghĩa giữa query và chunk.

**`search_with_filter`** — approach:
    Thực hiện lọc metadata trước khi truy vấn vector. Ví dụ:

    metadata_filter={"source": "alzheimer"}

    Giúp:
    - Tăng precision
    - Nhưng có thể giảm recall nếu filter quá chặt
### Agent
#### RAG: retrieve → prompt → LLM
**`answer`** — approach:

    Triển khai mô hình RAG theo pipeline:

    - Retrieve top-k chunks từ vector store
    - Ghép thành context
    - Đưa vào prompt cùng câu hỏi
    - Gọi LLM (GitHub API)

**`Prompt được thiết kế theo dạng:`**

    Context:
    [Chunk 1]
    [Chunk 2]

    Question: ...
    Answer:

**`Có thêm ràng buộc:`**

    Chỉ trả lời dựa trên context
    → giảm hallucination

---

### Test Results
 
**`test session starts `**

collected 42 items                                                                                                                                             

tests\test_solution.py ..........................................                                                                                        [100%]

===================================================================== 42 passed in 4.13s ======================================================================
## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Bệnh nhân bị sốt cao và đau bụng vùng hạ sườn phải | Triệu chứng phổ biến nhất là sốt và đau ở hạ sườn  | High | 0.7565 | Yes |
| 2 | Amip xâm nhập vào cơ thể qua niêm mạc mũi. | Ký sinh trùng đi vào người bằng đường hô hấp qua m | Low | 0.4150 | No |
| 3 | Lãi suất tiền gửi tiết kiệm đang giảm mạnh. | Tỷ lệ lạm phát đang có dấu hiệu hạ nhiệt. | High | 0.5695 | No |
| 4 | Cần phẫu thuật để nạo vét hết các ổ mủ. | Điều trị bằng thuốc kháng sinh liều cao. | High | 0.5640 | No |
| 5 | Nước tiểu sẽ chuyển sang màu đen khi để ngoài khôn | Màu sắc của nước tiểu thay đổi do phản ứng oxy hóa | High | 0.7219 | Yes |


**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

    Kết quả bất ngờ nhất là Pair 3 và Pair 4.

    Pair 3: Hai câu khác domain (tài chính vs kinh tế) nhưng vẫn bị dự đoán High.
    Pair 4: Hai phương pháp điều trị khác nhau (phẫu thuật vs thuốc) nhưng vẫn bị xem là tương đồng.

    -> Điều này cho thấy embeddings không chỉ dựa vào từ khóa mà còn học các pattern ngữ nghĩa chung, nên đôi khi coi các câu “cùng bối cảnh” là giống nhau dù ý nghĩa thực tế khác.

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers 

=== RESULTS ===

Q1: Bệnh Amip ăn não lây qua đường nào và ủ bệnh bao lâu?
Relevant: False
Correct: False
Answer: bệnh amip ăn não không lây từ người sang người hoặc từ người bệnh sang nước. amip naegleria thường xâm nhập vào cơ thể q...

Q2: Nguyên nhân áp xe hậu môn tái phát?
Relevant: False
Correct: False
Answer: nguyên nhân áp xe hậu môn tái phát thường do việc điều trị ban đầu không triệt để, dẫn đến mủ và vi khuẩn còn sót lại tr...

Q3: Tam chứng Fontan áp xe gan gồm gì?
Relevant: False
Correct: True
Answer: tam chứng fontan trong áp xe gan gồm ba dấu hiệu chính sau:

1. sốt cao kéo dài  
2. đau vùng hạ sườn phải  
3. gan to v...

Q4: Alkapton niệu do gen nào?
Relevant: False
Correct: False
Answer: alkapton niệu do đột biến ở gen hgd (gen encoding enzyme homogentisate 1,2-dioxygenase) gây ra. đột biến này làm thiếu h...

Q5: Điều trị ám ảnh sợ hãi?
Relevant: False
Correct: False
Answer: điều trị ám ảnh sợ hãi có thể bao gồm nhiều phương pháp khác nhau. ngoài các phương pháp điều trị truyền thống, một số p...

--- SUMMARY ---
Relevant Top-3: 0/5
Correct Answer: 1/5

### Nhận xét

- Hệ thống có hiệu suất retrieval rất thấp: 0/5 queries có chunk relevant trong top-3.
- Điều này cho thấy vector search chưa truy xuất đúng nội dung liên quan đến câu hỏi.

- Tuy nhiên, agent vẫn trả lời đúng 1/5 câu hỏi (Tam chứng Fontan), chứng tỏ LLM có khả năng suy luận hoặc “đoán” từ kiến thức nền, dù không có context phù hợp.

- Một số câu trả lời đúng về mặt nội dung (ví dụ câu 4: gen HGD) nhưng vẫn bị đánh giá sai do không match đúng format keyword trong benchmark.

-> Kết luận: Pipeline hiện tại gặp vấn đề nghiêm trọng ở bước retrieval (vector search), dẫn đến việc agent không được cung cấp đúng context để trả lời.

---
**Bao nhiêu queries trả về chunk relevant trong top-3?** 0 / 5

## 7. What I Learned (5 điểm)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

    Tôi nhận ra rằng việc tuning tham số `chunk_size` và `overlap` có ảnh hưởng trực tiếp đến chất lượng retrieval. Chunk quá nhỏ làm mất context, trong khi chunk quá lớn làm giảm precision.

 **Điều hay nhất tôi học được từ nhóm khác (qua demo):**

    Việc sử dụng metadata filtering trước khi truy vấn vector giúp giảm đáng kể nhiễu trong kết quả, đặc biệt khi làm việc với nhiều tài liệu thuộc các domain khác nhau.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

    Tôi sẽ áp dụng hybrid search (kết hợp BM25 và vector search) để cải thiện khả năng tìm kiếm các từ khóa cụ thể như "tái phát", đồng thời thiết kế metadata chi tiết hơn (ví dụ: disease, symptom, treatment) để tăng hiệu quả filtering.

---

## Tự đánh giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 12 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **86 / 100** |

