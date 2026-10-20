import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Cấu hình giao diện trang web
st.set_page_config(page_title="My Custom AI", page_icon="🤖", layout="centered")
st.title("🤖 Trò chuyện với AI Tự Chế từ PyTorch")
st.caption("Mô hình Transformer được code và huấn luyện hoàn toàn từ con số 0!")

# 1. Định nghĩa kiến trúc mô hình PyTorch thuần túy
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

# 2. Khởi tạo và huấn luyện nhanh mô hình ngay khi web khởi động
@st.cache_resource
def load_and_train_ai():
    raw_text = """
    Xin chào, tôi là AI tự chế. AI này được lập trình hoàn toàn bằng Python và PyTorch. 
    Hỏi: Bạn tên là gì?
    Đáp: Tôi là mô hình AI tự chế được huấn luyện từ con số 0.
    Hỏi: Bạn được làm bằng gì?
    Đáp: Tôi được xây dựng bằng Python và thư viện PyTorch thuần túy.
    Hỏi: Thời tiết hôm nay thế nào?
    Đáp: Hôm nay trời rất đẹp và thuận lợi để lập trình AI.
    Hỏi: Bạn có thông minh không?
    Đáp: Tôi đang học hỏi từng ngày từ dữ liệu bạn cung cấp.
    Hỏi: Lập trình có vui không?
    Đáp: Lập trình rất thú vị khi bạn tự tay xây dựng mọi thứ từ đầu.
    """
    
    # Huấn luyện Tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer_tk = trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator([raw_text], trainer_tk)
    
    # Khởi tạo mô hình
    vocab_size = 600
    model = MyCustomAI(vocab_size=vocab_size)
    
    # Huấn luyện nhanh mô hình
    encoded = tokenizer.encode(raw_text).ids
    block_size = 16
    data = [encoded[i:i + block_size + 1] for i in range(0, len(encoded) - block_size)]
    
    if data:
        inputs = torch.tensor([d[:-1] for d in data])
        targets = torch.tensor([d[1:] for d in data])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for _ in range(40):
            optimizer.zero_grad()
            loss = criterion(model(inputs).view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model, tokenizer

with st.spinner("Đang khởi động và huấn luyện AI... Chờ xíu nhé! 🚀"):
    model, tokenizer = load_and_train_ai()

st.success("✨ AI đã sẵn sàng trò chuyện!")

# 3. Quản lý lịch sử khung chat trên giao diện
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Nhận câu hỏi từ người dùng qua ô chat bên dưới
if prompt := st.chat_input("Nhập câu hỏi (ví dụ: Hỏi: Bạn tên là gì?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        encoded_start = tokenizer.encode(prompt).ids
        if not encoded_start:
            encoded_start = [0]
        x = torch.tensor([encoded_start])

        with torch.no_grad():
            for _ in range(35):
                logits = model(x)
                next_token = torch.argmax(logits[0, -1, :]).item()
                encoded_start.append(next_token)
                x = torch.tensor([encoded_start])

        ai_response = tokenizer.decode(encoded_start)
        st.markdown(ai_response)
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response})