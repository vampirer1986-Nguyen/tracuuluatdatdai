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

## 2. Lấy API Key

- **Pinecone**: đăng ký miễn phí tại https://app.pinecone.io, vào mục
  "API Keys" để lấy khóa. Lưu ý dự án Pinecone cần được tạo ở khu vực hỗ
  trợ mô hình `llama-text-embed-v2` (mặc định ứng dụng dùng vùng
  `aws` / `us-east-1`).
- **Groq**: đăng ký miễn phí tại https://console.groq.com, vào mục
  "API Keys" để tạo khóa.

Bạn **không cần chỉnh sửa code** — chỉ cần nhập 2 khóa trên vào thanh bên
trái (sidebar) khi mở ứng dụng.

## 3. Chạy ứng dụng

```bash
streamlit run app.py
```

Sau đó mở trình duyệt theo địa chỉ Streamlit hiển thị (thường là
http://localhost:8501).

## 4. Sử dụng

1. Nhập Pinecone API Key và Groq API Key ở sidebar.
2. Ở mục "1. Tải lên văn bản PDF", chọn một hoặc nhiều file PDF rồi bấm
   "Tải lên & Lưu vào Pinecone". File đã lưu trước đó (trùng tên) sẽ tự
   động bị bỏ qua.
3. Ở mục "2. Đặt câu hỏi", gõ câu hỏi và bấm "Tìm câu trả lời". Ứng dụng sẽ
   hiển thị câu trả lời kèm nguồn tham khảo (tên file, đoạn văn bản).

## 5. Deploy lên Streamlit Community Cloud (miễn phí)

1. Đẩy code lên một repository GitHub (repo có thể public hoặc private).
2. Vào https://share.streamlit.io, đăng nhập bằng tài khoản GitHub.
3. Bấm "New app", chọn repo/branch vừa đẩy lên, chọn file chính là `app.py`.
4. (Tuỳ chọn) Vào "Advanced settings" → "Secrets", dán nội dung theo mẫu
   trong file [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)
   và điền API Key thật của bạn vào. Khi đó app sẽ tự điền sẵn API Key mỗi
   lần bạn mở lên (vẫn có thể sửa lại ở sidebar nếu muốn dùng key khác).
5. Bấm "Deploy". Sau vài phút app sẽ có link dạng
   `https://ten-app.streamlit.app` để bạn truy cập từ bất kỳ đâu.

Lưu ý: file `.streamlit/secrets.toml` (chứa key thật) đã được thêm vào
`.gitignore` — không bao giờ bị đẩy lên GitHub.

## Ghi chú kỹ thuật (không bắt buộc đọc)

- Index Pinecone được tạo tự động ở lần chạy đầu tiên (serverless, dùng
  "integrated inference" nên Pinecone tự embedding văn bản bằng
  `llama-text-embed-v2`, không cần gọi API embedding riêng ở phía ứng dụng).
- PDF được cắt thành các đoạn ~1200 ký tự, chồng lấn 150 ký tự, để tìm kiếm
  chính xác hơn.
- Việc kiểm tra trùng file dựa trên mã băm (hash) của tên file, dùng làm
  tiền tố (prefix) của các ID vector trong Pinecone.
