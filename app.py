import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Cấu hình giao diện
st.set_page_config(
    page_title="NemoAI Workspace",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/chatbot.png", width=100)
    st.markdown("### ⚙️ Thông số Mô hình")
    st.info(
        "**Framework:** PyTorch\n\n"
        "**Architecture:** TransformerEncoder\n\n"
        "**Tokenizer:** BPE\n\n"
        "**Status:** Đang hoạt động ổn định"
    )
    st.divider()
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 NemoAI Workspace")
st.caption("Mô hình AI tự chế chạy trên nền tảng PyTorch.")

# 3. Kiến trúc mô hình tối ưu (An toàn cho RAM)
class StableCustomAI(nn.Module):
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

# 4. Huấn luyện an toàn
@st.cache_resource
def load_and_train_ai():
    raw_text = """
    Hỏi: Bạn tên là gì? Đáp: Tôi là NemoAI, trí tuệ nhân tạo tự chế.
    Hỏi: Bạn được tạo ra bởi ai? Đáp: Tôi được lập trình bởi bạn.
    Hỏi: Bạn làm bằng ngôn ngữ gì? Đáp: Tôi viết bằng Python và PyTorch.
    Hỏi: Thời tiết hôm nay thế nào? Đáp: Hôm nay rất đẹp để code AI.
    Hỏi: 1 cộng 1 bằng mấy? Đáp: 1 cộng 1 bằng 2.
    Hỏi: Thủ đô Việt Nam là gì? Đáp: Thủ đô của Việt Nam là Hà Nội.
    """ * 10
    
    vocab_size = 500
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer_tk = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator([raw_text], trainer_tk)
    
    model = StableCustomAI(vocab_size=vocab_size)
    
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

with st.spinner("Đang khởi tạo hệ thống..."):
    model, tokenizer = load_and_train_ai()

# 5. Lịch sử chat & Giao diện
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi đã sẵn sàng trò chuyện cùng bạn."}
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
            
        st.session_state.messages.append({"role": "assistant", "content": ai_response})import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Cấu hình giao diện trang web
st.set_page_config(
    page_title="NemoAI - Autonomous PyTorch Model",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Giao diện CSS tùy chỉnh
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar thông số
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/chatbot.png", width=100)
    st.markdown("### ⚙️ Thông số Mô hình")
    st.info(
        "**Framework:** PyTorch\n\n"
        "**Architecture:** TransformerEncoder (Deep)\n\n"
        "**Tokenizer:** BPE\n\n"
        "**Status:** Đã mở rộng bộ nhớ & dữ liệu"
    )
    st.divider()
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 NemoAI Workspace")
st.caption("Mô hình AI tự chế được huấn luyện với tập dữ liệu mở rộng.")

# 4. Kiến trúc mô hình lớn hơn (4 lớp Transformer, chiều nhúng 128)
class AdvancedCustomAI(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256, batch_first=True
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

# 5. Kho dữ liệu huấn luyện mở rộng
@st.cache_resource
def load_and_train_ai():
    raw_text = """
    Hỏi: Bạn tên là gì? Đáp: Tôi là NemoAI, trí tuệ nhân tạo tự chế từ PyTorch.
    Hỏi: Bạn được tạo ra bởi ai? Đáp: Tôi được lập trình và phát triển bởi bạn.
    Hỏi: Bạn làm bằng ngôn ngữ gì? Đáp: Tôi được viết hoàn toàn bằng Python và PyTorch.
    Hỏi: Thời tiết hôm nay thế nào? Đáp: Hôm nay trời rất đẹp, thích hợp để viết code AI.
    Hỏi: 1 cộng 1 bằng mấy? Đáp: 1 cộng 1 bằng 2.
    Hỏi: Thủ đô Việt Nam là gì? Đáp: Thủ đô của Việt Nam là Hà Nội.
    Hỏi: Bạn có thông minh không? Đáp: Tôi đang học hỏi và nâng cấp từng ngày.
    Hỏi: Chào bạn. Đáp: Xin chào! Tôi có thể giúp gì cho bạn hôm nay?
    Hỏi: Bạn có thích lập trình không? Đáp: Tôi rất thích lập trình Python và mạng neuron.
    Hỏi: Máy tính là gì? Đáp: Máy tính là thiết bị điện tử giúp xử lý thông tin và tính toán.
    """ * 5 # Nhân bản dữ liệu để mô hình dễ ghi nhớ hơn trong quá trình train nhanh
    
    vocab_size = 1000
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer_tk = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator([raw_text], trainer_tk)
    
    model = AdvancedCustomAI(vocab_size=vocab_size)
    
    encoded = tokenizer.encode(raw_text).ids
    block_size = 32
    data = [encoded[i:i + block_size + 1] for i in range(0, len(encoded) - block_size)]
    
    if data:
        inputs = torch.tensor([d[:-1] for d in data])
        targets = torch.tensor([d[1:] for d in data])
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for _ in range(80): # Tăng số vòng học lên 80 vòng
            optimizer.zero_grad()
            loss = criterion(model(inputs).view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model, tokenizer

with st.spinner("Đang huấn luyện mô hình với tập dữ liệu mở rộng..."):
    model, tokenizer = load_and_train_ai()

# 6. Quản lý lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI phiên bản nâng cấp. Hãy hỏi tôi bất cứ điều gì trong danh mục dữ liệu nhé!"}
    ]

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 7. Khung nhập liệu
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
                for _ in range(40):
                    logits = model(x)
                    next_token = torch.argmax(logits[0, -1, :]).item()
                    encoded_start.append(next_token)
                    x = torch.tensor([encoded_start])

            ai_response = tokenizer.decode(encoded_start)
            st.markdown(ai_response)
            
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
