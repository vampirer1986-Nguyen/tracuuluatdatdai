# Tra cứu Luật Đất Đai Việt Nam (RAG: Pinecone + Groq)

Ứng dụng Streamlit cho phép:
- Tải lên các file PDF (luật, nghị định, thông tư...), tự động đọc nội dung
  và lưu vào Pinecone (index `luat-dat-dai-vn`), dùng mô hình embedding tích
  hợp sẵn của Pinecone là `llama-text-embed-v2` (không cần tự gọi API
  embedding riêng).
- File trùng tên đã tồn tại trong Pinecone sẽ tự động được bỏ qua khi tải lên.
- Nhập câu hỏi, ứng dụng tìm các đoạn văn bản liên quan trong Pinecone rồi
  dùng Groq (mô hình Llama 3.3) để tạo câu trả lời dựa trên nội dung đó.

## 1. Cài đặt

```bash
pip install -r requirements.txt
```

## 2. Lấy API Key và cấu hình (KHÔNG nhập trên giao diện)

- **Pinecone**: đăng ký miễn phí tại https://app.pinecone.io, vào mục
  "API Keys" để lấy khóa. Lưu ý dự án Pinecone cần được tạo ở khu vực hỗ
  trợ mô hình `llama-text-embed-v2` (mặc định ứng dụng dùng vùng
  `aws` / `us-east-1`).
- **Groq**: đăng ký miễn phí tại https://console.groq.com, vào mục
  "API Keys" để tạo khóa.

Vì app deploy công khai (ai có link cũng vào được), 2 key này **không được
nhập hay hiển thị trên giao diện** để tránh bị lộ cho người khác — thay vào
đó chỉ chủ app (bạn) mới cấu hình được, thông qua Secrets phía máy chủ:

- **Chạy local**: tạo file `.streamlit/secrets.toml` (copy từ file mẫu
  [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)) rồi
  điền key thật vào. File này đã có trong `.gitignore` nên sẽ không bao giờ
  bị đẩy lên GitHub.
- **Deploy trên Streamlit Cloud**: điền key thật vào mục Settings → Secrets
  của app (xem mục 5 bên dưới).

## 3. Chạy ứng dụng

```bash
streamlit run app.py
```

Sau đó mở trình duyệt theo địa chỉ Streamlit hiển thị (thường là
http://localhost:8501). Sidebar chỉ hiển thị trạng thái "đã cấu hình /
chưa cấu hình" cho từng key, không hiển thị giá trị key.

## 4. Sử dụng

1. Ở mục "1. Tải lên văn bản PDF", chọn một hoặc nhiều file PDF rồi bấm
   "Tải lên & Lưu vào Pinecone". File đã lưu trước đó (trùng tên) sẽ tự
   động bị bỏ qua.
2. Ở mục "2. Đặt câu hỏi", gõ câu hỏi và bấm "Tìm câu trả lời". Ứng dụng sẽ
   hiển thị câu trả lời kèm nguồn tham khảo (tên file, đoạn văn bản).

## 5. Deploy lên Streamlit Community Cloud (miễn phí)

1. Đẩy code lên một repository GitHub (repo có thể public hoặc private).
2. Vào https://share.streamlit.io, đăng nhập bằng tài khoản GitHub.
3. Bấm "New app", chọn repo/branch vừa đẩy lên, chọn file chính là `app.py`.
4. **Bắt buộc**: vào "Advanced settings" → "Secrets", dán nội dung theo mẫu
   trong file [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)
   và điền API Key thật của bạn vào. Không có bước này app sẽ báo lỗi "chưa
   cấu hình" và không dùng được.
5. Bấm "Deploy". Sau vài phút app sẽ có link dạng
   `https://ten-app.streamlit.app` để bạn truy cập từ bất kỳ đâu — người
   khác mở link cũng dùng được app (bằng key của bạn) nhưng không thể xem
   hay lấy được key đó.

⚠️ Vì bất kỳ ai có link cũng dùng được key của bạn để gọi Pinecone/Groq,
hãy cân nhắc thêm lớp mật khẩu bảo vệ app nếu không muốn người lạ dùng
chung quota, hoặc theo dõi usage trên dashboard Pinecone/Groq.

## 6. Chạy bằng Docker

```bash
docker build -t tracuuluatdatdai .
docker run -d -p 8501:8501 \
  -e PINECONE_API_KEY="key-thật-của-bạn" \
  -e GROQ_API_KEY="key-thật-của-bạn" \
  --name tracuuluatdatdai \
  tracuuluatdatdai
```

Sau đó mở http://localhost:8501. Key được truyền qua biến môi trường
(`-e`), không cần mount file secrets. Muốn dừng: `docker stop tracuuluatdatdai`.

## Ghi chú kỹ thuật (không bắt buộc đọc)

- Index Pinecone được tạo tự động ở lần chạy đầu tiên (serverless, dùng
  "integrated inference" nên Pinecone tự embedding văn bản bằng
  `llama-text-embed-v2`, không cần gọi API embedding riêng ở phía ứng dụng).
- PDF được cắt thành các đoạn ~1200 ký tự, chồng lấn 150 ký tự, để tìm kiếm
  chính xác hơn.
- Việc kiểm tra trùng file dựa trên mã băm (hash) của tên file, dùng làm
  tiền tố (prefix) của các ID vector trong Pinecone.
