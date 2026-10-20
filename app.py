import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Cấu hình trang web (Chế độ rộng, tiêu đề chuẩn AI)
st.set_page_config(
    page_title="NemoAI - Autonomous PyTorch Model",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Tùy chỉnh giao diện CSS cho giống các App AI cao cấp
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
    h1 {
        color: #1E1E1E;
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar (Thanh công cụ bên trái)
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/chatbot.png", width=100)
    st.markdown("### ⚙️ Thông số Mô hình")
    st.info(
        "**Framework:** PyTorch\n\n"
        "**Architecture:** TransformerEncoder\n\n"
        "**Tokenizer:** BPE (Whitespace)\n\n"
        "**Status:** 100% Autonomous & Proprietary"
    )
    
    st.divider()
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.caption("Được xây dựng và làm chủ hoàn toàn bởi bạn! 🚀")

# Tiêu đề chính giao diện
st.title("🤖 NemoAI Workspace")
st.caption("Trò chuyện trực tiếp với mô hình Transformer tự code từ con số 0.")

# 4. Định nghĩa kiến trúc PyTorch
class MyCustomAI(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=2, num_layers=2, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.embed(idx) + self.pos_embed(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(idx.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.fc_out(x)

# 5. Khởi tạo và huấn luyện mô hình khi khởi động web
@st.cache_resource
def load_and_train_ai():
    raw_text = """
    Xin chào, tôi là NemoAI tự chế. Tôi được lập trình hoàn toàn bằng Python và PyTorch. 
    Hỏi: Bạn tên là gì?
    Đáp: Tôi là mô hình AI tự chế được huấn luyện từ con số 0.
    Hỏi: Bạn được làm bằng gì?
    Đáp: Tôi được xây dựng bằng Python và thư viện PyTorch thuần túy.
    Hỏi: Thời tiết hôm nay thế nào?
    Đáp: Hôm nay trời rất đẹp và thuận lợi để lập trình AI.
    Hỏi: Bạn có thông minh không?
    Đáp: Tôi đang học hỏi từng ngày từ dữ liệu bạn cung cấp.
    """
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer_tk = trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator([raw_text], trainer_tk)
    
    vocab_size = 600
    model = MyCustomAI(vocab_size=vocab_size)
    
    encoded = tokenizer.encode(raw_text).ids
    block_size = 16
    data = [encoded[i:i + block_size + 1] for i in range(0, len(encoded) - block_size)]
    
    if data:
        inputs = torch.tensor([d[:-1] for d in data])
        targets = torch.tensor([d[1:] for d in data])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for _ in range(35):
            optimizer.zero_grad()
            loss = criterion(model(inputs).view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model, tokenizer

with st.spinner("Đang khởi động hệ thống thần kinh nhân tạo..."):
    model, tokenizer = load_and_train_ai()

# 6. Quản lý lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI. Hãy đặt câu hỏi cho tôi nhé!"}
    ]

# Hiển thị lịch sử tin nhắn với avatar chuyên nghiệp
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 7. Khung nhập liệu chat ở đáy màn hình
if prompt := st.chat_input("Nhập câu hỏi (ví dụ: Hỏi: Bạn tên là gì?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("AI đang suy nghĩ..."):
            encoded_start = tokenizer.encode(prompt).ids
            if not encoded_start:
                encoded_start = [0]
            x = torch.tensor([encoded_start])

            with torch.no_grad():
                for _ in range(30):
                    logits = model(x)
                    next_token = torch.argmax(logits[0, -1, :]).item()
                    encoded_start.append(next_token)
                    x = torch.tensor([encoded_start])

            ai_response = tokenizer.decode(encoded_start)
            st.markdown(ai_response)
            
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
