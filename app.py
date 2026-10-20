import streamlit as st
from google import genai

# 1. Cấu hình giao diện trang Streamlit
st.set_page_config(
    page_title="NemoAI Pro Workspace",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Quản lý API Key
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.title("⚙️ Cấu hình NemoAI")
    
    if not api_key:
        api_key = st.text_input("Nhập Google Gemini API Key:", type="password")
        st.markdown("[👉 Lấy Gemini API Key miễn phí](https://aistudio.google.com/)")
    else:
        st.success("✅ Đã kết nối API Key từ Secrets!")

    st.divider()
    
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 3. Tiêu đề ứng dụng
st.title("🤖 NemoAI Pro")
st.caption("Trợ lý AI thông minh, phản hồi thời gian thực trên đám mây.")

# 4. Kiểm tra API Key
if not api_key:
    st.info("💡 Vui lòng nhập Gemini API Key ở thanh bên (Sidebar) để bắt đầu trò chuyện!")
    st.stop()

# Khởi tạo Client
try:
    client = genai.Client(api_key=api_key.strip())
except Exception as e:
    st.error(f"⚠️ Lỗi kết nối Client: {e}")
    st.stop()

# 5. Khởi tạo danh sách lưu trữ lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI. Tôi có thể giúp gì cho bạn hôm nay?"}
    ]

# 6. Hiển thị lại lịch sử tin nhắn
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 7. Xử lý khi gửi tin nhắn mới
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("NemoAI đang suy nghĩ..."):
            try:
                # Gọi trực tiếp mô hình gemini-1.5-flash
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=prompt,
                )
                ai_response = response.text
                
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"⚠️ Đã xảy ra lỗi kết nối: {e}")
