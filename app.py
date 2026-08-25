"""
Ứng dụng tra cứu Luật Đất Đai Việt Nam (RAG với Pinecone + Groq)
==================================================================

Cách hoạt động (tự động, không cần biết kỹ thuật):
1. Bạn tải lên các file PDF (ví dụ: Luật Đất đai, Nghị định, Thông tư...).
2. Ứng dụng đọc nội dung PDF, cắt thành các đoạn nhỏ (chunk) và lưu vào
   Pinecone (index "luat-dat-dai-vn"). Việc "embedding" (chuyển văn bản
   thành vector) được Pinecone tự làm bằng mô hình "llama-text-embed-v2"
   (integrated inference) — bạn không cần gọi API embedding riêng.
3. File đã tải lên rồi (trùng tên) sẽ tự động được bỏ qua, không lưu lại.
4. Bạn gõ câu hỏi vào ô tìm kiếm, ứng dụng sẽ:
   - Tìm các đoạn văn bản liên quan nhất trong Pinecone.
   - Gửi các đoạn đó + câu hỏi cho Groq (mô hình Llama 3.3) để tạo câu trả lời.
"""

import hashlib
import time

import pymupdf
import streamlit as st

# ----------------------------------------------------------------------------
# Cấu hình chung
# ----------------------------------------------------------------------------
INDEX_NAME = "luat-dat-dai-vn"
EMBED_MODEL = "llama-text-embed-v2"
NAMESPACE = "default"
TEXT_FIELD = "chunk_text"          # tên field chứa văn bản, dùng cho embedding
GROQ_MODEL_FALLBACK = "openai/gpt-oss-120b"  # dùng nếu không lấy được danh sách model từ Groq
# Model không dùng để trả lời (nhận dạng giọng nói, kiểm duyệt, chuyển văn bản->giọng nói...)
GROQ_EXCLUDE_PATTERNS = ("whisper", "tts", "guard", "moderation")
CLOUD = "aws"
REGION = "us-east-1"
CHUNK_SIZE = 1200                  # số ký tự mỗi đoạn
CHUNK_OVERLAP = 150                # số ký tự gối lên nhau giữa các đoạn
TOP_K = 6                          # số đoạn liên quan lấy ra khi tìm kiếm
UPSERT_BATCH = 90                  # Pinecone giới hạn ~96 record/lần upsert_records

st.set_page_config(page_title="Tra cứu Luật Đất Đai VN", page_icon="⚖️", layout="wide")


# ----------------------------------------------------------------------------
# Tiện ích: đọc PDF, cắt đoạn văn bản
# ----------------------------------------------------------------------------
def extract_text_from_pdf(file) -> str:
    file_bytes = file.read()
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    try:
        pages_text = [page.get_text() for page in doc]
    finally:
        doc.close()
    return "\n".join(pages_text)


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())  # gộp khoảng trắng thừa
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks


def file_hash(filename: str) -> str:
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Pinecone: khởi tạo, kiểm tra/ tạo index, upsert, tìm kiếm
# ----------------------------------------------------------------------------
def get_pinecone_client(api_key: str):
    from pinecone import Pinecone
    return Pinecone(api_key=api_key)


def ensure_index(pc, index_name: str):
    """Tạo index với embedding tích hợp sẵn nếu chưa tồn tại."""
    existing = pc.list_indexes().names()
    if index_name not in existing:
        pc.create_index_for_model(
            name=index_name,
            cloud=CLOUD,
            region=REGION,
            embed={
                "model": EMBED_MODEL,
                "field_map": {"text": TEXT_FIELD},
            },
        )
        # Chờ index sẵn sàng
        while True:
            desc = pc.describe_index(index_name)
            if desc.status.get("ready"):
                break
            time.sleep(1)
    return pc.Index(index_name)


def file_already_exists(index, namespace: str, fhash: str) -> bool:
    try:
        for page in index.list(prefix=f"{fhash}::", namespace=namespace):
            if page.vectors:
                return True
        return False
    except Exception:
        return False


def list_stored_files(index, namespace: str):
    """Trả về danh sách (tên file) đã lưu, bằng cách gom các id theo prefix
    rồi lấy metadata của 1 id đại diện cho mỗi file."""
    seen_prefixes = set()
    files = []
    try:
        for page in index.list(namespace=namespace):
            for item in page.vectors:
                vid = item.id
                prefix = vid.split("::")[0]
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                try:
                    fetched = index.fetch(ids=[vid], namespace=namespace)
                    v = fetched.vectors.get(vid)
                    meta = v.metadata if v else None
                    if meta and meta.get("filename"):
                        files.append(meta["filename"])
                except Exception:
                    continue
    except Exception:
        pass
    return sorted(set(files))


def upsert_pdf(index, namespace: str, filename: str, chunks: list[str]):
    fhash = file_hash(filename)
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "_id": f"{fhash}::{i:04d}",
            TEXT_FIELD: chunk,
            "filename": filename,
            "chunk_index": i,
        })
    for i in range(0, len(records), UPSERT_BATCH):
        batch = records[i:i + UPSERT_BATCH]
        index.upsert_records(namespace=namespace, records=batch)


def search_relevant_chunks(index, namespace: str, question: str, top_k: int = TOP_K):
    results = index.search_records(
        namespace=namespace,
        query={"inputs": {"text": question}, "top_k": top_k},
        fields=[TEXT_FIELD, "filename", "chunk_index"],
    )
    hits = results.result.hits
    parsed = []
    for hit in hits:
        fields = hit.fields
        parsed.append({
            "text": fields.get(TEXT_FIELD, ""),
            "filename": fields.get("filename", "?"),
            "chunk_index": fields.get("chunk_index", 0),
            "score": hit.score,
        })
    return parsed


# ----------------------------------------------------------------------------
# Groq: sinh câu trả lời dựa trên các đoạn văn bản tìm được
# ----------------------------------------------------------------------------
def list_groq_chat_models(groq_api_key: str) -> list[str]:
    """Lấy danh sách model hiện đang khả dụng trên tài khoản Groq, loại bỏ
    các model không phải dùng để trả lời hội thoại (whisper, tts, guard...)."""
    from groq import Groq
    client = Groq(api_key=groq_api_key)
    response = client.models.list()
    ids = [
        m.id for m in response.data
        if not any(p in m.id.lower() for p in GROQ_EXCLUDE_PATTERNS)
    ]
    return sorted(ids)


def pick_default_groq_model(model_ids: list[str]) -> str:
    if not model_ids:
        return GROQ_MODEL_FALLBACK
    if GROQ_MODEL_FALLBACK in model_ids:
        return GROQ_MODEL_FALLBACK
    for m in model_ids:
        if "versatile" in m.lower():
            return m
    for m in model_ids:
        if "120b" in m or "70b" in m:
            return m
    return model_ids[0]


def ask_groq(groq_api_key: str, model: str, question: str, contexts: list[dict]) -> str:
    from groq import Groq
    client = Groq(api_key=groq_api_key)

    context_text = "\n\n".join(
        f"[Nguồn: {c['filename']} - đoạn {c['chunk_index']}]\n{c['text']}"
        for c in contexts
    )

    system_prompt = (
        "Bạn là trợ lý pháp lý, trả lời câu hỏi về Luật Đất đai Việt Nam. "
        "Chỉ dựa vào NỘI DUNG THAM KHẢO được cung cấp bên dưới để trả lời. "
        "Nếu nội dung tham khảo không đủ để trả lời, hãy nói rõ là không tìm thấy "
        "thông tin liên quan trong tài liệu đã tải lên, không tự bịa ra thông tin. "
        "Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng, có thể trích dẫn điều/khoản nếu có."
    )
    user_prompt = f"NỘI DUNG THAM KHẢO:\n{context_text}\n\nCÂU HỎI: {question}"

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content


def get_secret(key: str) -> str:
    """Đọc giá trị từ Streamlit Secrets nếu có cấu hình sẵn (khi deploy lên
    Streamlit Cloud), nếu không thì trả về chuỗi rỗng để người dùng tự nhập."""
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# Giao diện Streamlit
# ----------------------------------------------------------------------------
st.title("⚖️ Tra cứu Luật Đất Đai Việt Nam")
st.caption("Tải lên văn bản PDF, sau đó đặt câu hỏi để tra cứu nội dung liên quan.")

with st.sidebar:
    st.header("🔑 Cấu hình API")
    pinecone_api_key = st.text_input(
        "Pinecone API Key", type="password",
        value=st.session_state.get("pinecone_api_key") or get_secret("PINECONE_API_KEY"))
    groq_api_key = st.text_input(
        "Groq API Key", type="password",
        value=st.session_state.get("groq_api_key") or get_secret("GROQ_API_KEY"))
    st.session_state["pinecone_api_key"] = pinecone_api_key
    st.session_state["groq_api_key"] = groq_api_key

    groq_model = None
    if groq_api_key:
        @st.cache_data(show_spinner="Đang tải danh sách mô hình Groq...", ttl=3600)
        def _cached_groq_models(api_key: str):
            return list_groq_chat_models(api_key)

        try:
            model_options = _cached_groq_models(groq_api_key)
        except Exception as e:
            model_options = []
            st.warning(f"Không lấy được danh sách mô hình từ Groq ({e}). "
                       f"Sẽ thử dùng mô hình mặc định: `{GROQ_MODEL_FALLBACK}`.")

        if model_options:
            default_model = pick_default_groq_model(model_options)
            groq_model = st.selectbox(
                "Mô hình trả lời (Groq)",
                options=model_options,
                index=model_options.index(default_model),
            )
        else:
            groq_model = GROQ_MODEL_FALLBACK

    st.session_state["groq_model"] = groq_model

    st.markdown("---")
    st.caption(f"Index Pinecone: `{INDEX_NAME}`")
    st.caption(f"Mô hình embedding: `{EMBED_MODEL}`")

if not pinecone_api_key:
    st.info("👈 Vui lòng nhập Pinecone API Key ở thanh bên trái để bắt đầu.")
    st.stop()

# Khởi tạo Pinecone / index (cache theo API key để không tạo lại mỗi lần)
@st.cache_resource(show_spinner="Đang kết nối tới Pinecone...")
def _init_index(api_key: str):
    pc = get_pinecone_client(api_key)
    return ensure_index(pc, INDEX_NAME)

try:
    index = _init_index(pinecone_api_key)
except Exception as e:
    st.error(f"Không thể kết nối / tạo index Pinecone: {e}")
    st.stop()

# ----------------------------------------------------------------------------
# Phần 1: Tải lên PDF
# ----------------------------------------------------------------------------
st.subheader("1. Tải lên văn bản PDF")

uploaded_files = st.file_uploader(
    "Chọn một hoặc nhiều file PDF", type=["pdf"], accept_multiple_files=True
)

if "last_upload_summary" not in st.session_state:
    st.session_state["last_upload_summary"] = None

if uploaded_files and st.button("📤 Tải lên & Lưu vào Pinecone", type="primary"):
    progress = st.progress(0, text="Đang xử lý...")
    total = len(uploaded_files)
    added, skipped, failed = [], [], []

    for i, f in enumerate(uploaded_files):
        progress.progress(i / total, text=f"Đang xử lý: {f.name}")
        fhash = file_hash(f.name)
        try:
            if file_already_exists(index, NAMESPACE, fhash):
                skipped.append(f.name)
                continue
            text = extract_text_from_pdf(f)
            chunks = split_into_chunks(text)
            if not chunks or len(text.strip()) < 200:
                failed.append((
                    f.name,
                    "Không trích xuất được nội dung văn bản (chỉ đọc được "
                    f"{len(text.strip())} ký tự). File có thể là bản scan/ảnh "
                    "không có lớp văn bản — hãy dùng công cụ OCR (ví dụ mở file "
                    "bằng Google Drive > Google Docs để tự động OCR) rồi tải "
                    "lại file đã OCR.",
                ))
                continue
            upsert_pdf(index, NAMESPACE, f.name, chunks)
            added.append((f.name, len(chunks)))
        except Exception as e:
            failed.append((f.name, str(e)))

    progress.progress(1.0, text="Hoàn tất")
    if added:
        # Pinecone cần vài giây để dữ liệu vừa upsert xuất hiện trong list()/search()
        time.sleep(2)
        st.cache_data.clear()
    st.session_state["last_upload_summary"] = {"added": added, "skipped": skipped, "failed": failed}

summary = st.session_state["last_upload_summary"]
if summary:
    if summary["added"]:
        st.success("Đã lưu: " + ", ".join(f"{name} ({n} đoạn)" for name, n in summary["added"]))
    if summary["skipped"]:
        st.warning("Bỏ qua (đã tồn tại trong Pinecone): " + ", ".join(summary["skipped"]))
    if summary["failed"]:
        for name, err in summary["failed"]:
            st.error(f"Lỗi khi xử lý {name}: {err}")

with st.expander("📂 Danh sách file đã lưu trong Pinecone", expanded=bool(summary and summary.get("added"))):
    if st.button("🔄 Làm mới danh sách"):
        st.cache_data.clear()

    @st.cache_data(show_spinner="Đang tải danh sách file...", ttl=30)
    def _list_files(_index):
        return list_stored_files(_index, NAMESPACE)

    files = _list_files(index)
    if files:
        for name in files:
            st.write(f"- {name}")
    else:
        st.caption("Chưa có file nào được lưu.")

# ----------------------------------------------------------------------------
# Phần 2: Đặt câu hỏi
# ----------------------------------------------------------------------------
st.subheader("2. Đặt câu hỏi")

question = st.text_input("Nhập câu hỏi của bạn về Luật Đất đai...", "")

if st.button("🔍 Tìm câu trả lời") and question.strip():
    if not groq_api_key:
        st.error("Vui lòng nhập Groq API Key ở thanh bên trái để có thể tạo câu trả lời.")
        st.stop()

    with st.spinner("Đang tìm kiếm nội dung liên quan..."):
        try:
            contexts = search_relevant_chunks(index, NAMESPACE, question)
        except Exception as e:
            st.error(f"Lỗi khi tìm kiếm trong Pinecone: {e}")
            st.stop()

    if not contexts:
        st.warning("Không tìm thấy nội dung liên quan. Hãy thử tải thêm tài liệu hoặc đổi cách hỏi.")
    else:
        with st.spinner("Đang tạo câu trả lời..."):
            try:
                model = st.session_state.get("groq_model") or GROQ_MODEL_FALLBACK
                answer = ask_groq(groq_api_key, model, question, contexts)
            except Exception as e:
                st.error(f"Lỗi khi gọi Groq API: {e}")
                st.stop()

        st.markdown("### 💬 Trả lời")
        st.write(answer)

        with st.expander("📎 Nguồn tham khảo"):
            for c in contexts:
                st.markdown(f"**{c['filename']}** (đoạn {c['chunk_index']}, độ liên quan: {c['score']:.3f})")
                st.caption(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
