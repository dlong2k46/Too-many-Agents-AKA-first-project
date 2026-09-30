# Research Agent (LangGraph Multi-Agent)

Agent nghiên cứu chuyên sâu sử dụng mô hình **Multi-Agent** điều phối bằng **LangGraph** + NVIDIA NIM (deepseek-v4.1-flash).

### Kiến trúc Đồ thị (Nodes):
1. **`researcher` (Node)**: Chuyên gia điều tra thông tin, tìm kiếm trên web (`ddgs`), tra cứu Wikipedia và cào sâu nội dung toàn văn bài viết bằng `trafilatura`.
2. **`tools` (Node)**: Thực thi các công cụ tra cứu (`search_tool`, `wiki_tool`, `fetch_url_tool`) và tích lũy dữ kiện.
3. **`critic_writer` (Node)**: Biên tập viên phản biện & Tổng hợp, đối chiếu loại bỏ mâu thuẫn và xuất bài báo cáo nghiên cứu có cấu trúc JSON sâu sắc.

## Cấu trúc project

```
.
├── .env.example             # Mẫu cấu hình — copy thành .env rồi điền giá trị thật
├── requirements.txt         # Thư viện chạy chương trình (langgraph, trafilatura, ddgs, wikipedia...)
├── main.py                  # Điểm vào chương trình (CLI)
├── node.py                  # Trực quan hóa sơ đồ đồ thị LangGraph (ASCII & Mermaid độc lập)
├── src/                     # Lõi mã nguồn chính của Research Agent
│   ├── __init__.py
│   ├── agent.py             # Đồ thị StateGraph LangGraph (Researcher ➔ Tools ➔ CriticWriter)
│   ├── config.py            # Nạp & kiểm tra cấu hình từ .env (MỘT nơi duy nhất)
│   ├── schemas.py           # Định dạng dữ liệu đầu ra (ResearchResponse)
│   ├── tools.py             # Tool: search_tool (ddgs), wiki_tool, fetch_url_tool (trafilatura)
│   ├── prompts.py           # System prompts cho Researcher và Critic/Writer
│   ├── output_parsing.py    # Phân tích JSON từ output thô của agent
│   ├── observability.py     # Ghi trace kỹ thuật dạng JSONL (mỗi sự kiện 1 dòng)
│   ├── session_manager.py   # Quản lý lưu trữ & nạp lại các phiên nghiên cứu
│   └── transcript.py        # Ghi lịch sử hội thoại ra Markdown kèm trích xuất nguồn
└── tests/                   # Unit test tự động
```


## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Mở .env, điền NVIDIA_API_KEY
```

## Chạy

Chế độ hỏi liên tục qua dòng lệnh:

```bash
python main.py
```

Xem nhanh sơ đồ kiến trúc Workflow của LangGraph (độc lập không cần API key):

```bash
python node.py
# Hoặc lưu sơ đồ Mermaid ra file:
python node.py --save
```


Chạy một câu hỏi rồi thoát (tiện cho script hoặc kiểm thử nhanh):

```bash
python main.py --query "Đánh giá xu hướng AI Agent năm 2026"
```

Xem danh sách các phiên nghiên cứu đã lưu:

```bash
python main.py --list-sessions
```

Tiếp tục một phiên nghiên cứu cũ (nạp lại ngữ cảnh câu hỏi cũ và ghi tiếp vào transcript):

```bash
python main.py --session <SESSION_ID>
```


Mỗi lần chạy tạo/cập nhật thư mục riêng trong `runs/<timestamp>-<id>/` chứa:
- `transcript.md` — lịch sử hội thoại & chi tiết tra cứu từ các nguồn, đọc được ngay, diff được qua Git.
- `session_state.json` — lưu trữ trạng thái ngữ cảnh hội thoại để resume phiên.
- `trace.jsonl` — log kỹ thuật từng bước nội bộ của agent (prompt, quyết định gọi tool, kết quả tool).



## Đọc lại trace kỹ thuật

```bash
python -c "
import pandas as pd
df = pd.read_json('runs/<tên-thư-mục>/trace.jsonl', lines=True)
print(df[['event', 'tool']].value_counts())
"
```

hoặc dùng `jq`:

```bash
jq -c 'select(.event == "agent_action")' runs/<tên-thư-mục>/trace.jsonl
```

## Quan sát nâng cao (tùy chọn): LangSmith

Điền các biến `LANGSMITH_*` trong `.env` (xem `.env.example`). Nếu tài
khoản của bạn không ở khu vực Mỹ, nhớ đặt đúng `LANGSMITH_ENDPOINT`
(ví dụ `https://apac.api.smith.langchain.com` cho khu vực APAC).
