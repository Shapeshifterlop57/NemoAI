import streamlit as st
from groq import Groq

# 1. Cấu hình trang Streamlit
st.set_page_config(page_title="NemoAI Pro", page_icon="🤖", layout="centered")

# 2. Lấy API Key từ Secrets hoặc Sidebar
api_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.title("⚙️ Cấu hình NemoAI")
    if not api_key:
        api_key = st.text_input("Nhập Groq API Key (gsk_...):", type="password")
    else:
        st.success("✅ Đã kết nối Groq API Key!")
        
    st.divider()
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 NemoAI Pro")
st.caption("Trợ lý AI phản hồi siêu tốc độ.")

if not api_key:
    st.info("💡 Vui lòng nhập Groq API Key ở thanh bên để bắt đầu!")
    st.stop()

# Khởi tạo Groq Client
try:
    client = Groq(api_key=api_key.strip())
except Exception as e:
    st.error(f"⚠️ Lỗi kết nối Groq Client: {e}")
    st.stop()

# 3. Lịch sử cuộc trò chuyện
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI. Tôi có thể giúp gì cho bạn?"}
    ]

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 4. Xử lý tin nhắn
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("NemoAI đang suy nghĩ..."):
            try:
                # TỰ ĐỘNG TÌM MODEL ĐANG HOẠT ĐỘNG
                # 1. Thử dùng model chuẩn mới nhất
                target_model = "llama-3.1-8b-instant"
                
                # 2. Gọi API tạo phản hồi
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    model=target_model,
                )
                
                ai_response = chat_completion.choices[0].message.content
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                # Nếu model trên lỗi, tự động quét danh sách model thực tế từ API
                try:
                    models_list = client.models.list()
                    available_models = [m.id for m in models_list.data if "llama" in m.id or "mixtral" in m.id]
                    
                    if available_models:
                        fallback_model = available_models[0]
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            model=fallback_model,
                        )
                        ai_response = chat_completion.choices[0].message.content
                        st.markdown(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    else:
                        st.error("⚠️ Không tìm thấy model nào khả dụng trong tài khoản Groq.")
                except Exception as ex:
                    st.error(f"⚠️ Lỗi kết nối API: {ex}")
