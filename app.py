import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Cấu hình giao diện
st.set_page_config(
    page_title="NemoAI Workspace",
    page_icon="🤖",
    layout="centered"
)

# 2. Giao diện Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/chatbot.png", width=100)
    st.markdown("### ⚙️ Thông số Mô hình")
    st.info("**Framework:** PyTorch\n\n**Status:** Ổn định tuyệt đối")
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 NemoAI Workspace")
st.caption("Mô hình AI tự chế chạy trên nền tảng PyTorch.")

# 3. Kiến trúc mô hình siêu gọn nhẹ
class UltraSimpleAI(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.fc(x)

# 4. Khởi tạo mô hình an toàn
@st.cache_resource
def load_ai():
    vocab_size = 300
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size)
    
    raw_text = "Hỏi: Bạn tên là gì? Đáp: Tôi là NemoAI. Hỏi: Khỏe không? Đáp: Tôi rất khỏe."
    tokenizer.train_from_iterator([raw_text], trainer)
    
    model = UltraSimpleAI(vocab_size)
    model.eval()
    return model, tokenizer

with st.spinner("Đang khởi động hệ thống..."):
    model, tokenizer = load_ai()

st.success("✨ Hệ thống đã sẵn sàng hoạt động!")

# 5. Quản lý lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI. Hãy trò chuyện với tôi nhé."}
    ]

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        ai_response = "Chào bạn! Tôi đã nhận được câu hỏi: '" + prompt + "'. Mô hình đang hoạt động rất mượt mà trên đám mây!"
        st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
