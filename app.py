import streamlit as st
import google.generativeai as genai

# 1. Cấu hình trang Streamlit
st.set_page_config(
    page_title="NemoAI Pro Workspace",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Quản lý API Key (Ưu tiên lấy từ st.secrets)
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/chatbot.png", width=100)
    st.title("⚙️ Cấu hình NemoAI")
    
    # Nếu chưa cài Secrets thì cho nhập trực tiếp trên Sidebar
    if not api_key:
        api_key = st.text_input("Nhập Google Gemini API Key:", type="password")
        st.markdown("[👉 Lấy Gemini API Key miễn phí](https://aistudio.google.com/)")
    else:
        st.success("✅ Đã kết nối API Key từ Secrets!")

    st.divider()
    
    # Nút xóa lịch sử chat
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 3. Tiêu đề ứng dụng
st.title("🤖 NemoAI Pro")
st.caption("Trợ lý AI thông minh, phản hồi thời gian thực trên đám mây.")

# 4. Kiểm tra API Key trước khi khởi chạy
if not api_key:
    st.info("💡 Vui lòng nhập Gemini API Key ở thanh bên (Sidebar) hoặc thêm vào Secrets để bắt đầu trò chuyện!")
    st.stop()

# Khởi tạo thư viện Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# 5. Khởi tạo lịch sử cuộc trò chuyện
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI. Tôi có thể giúp gì cho bạn hôm nay?"}
    ]

# 6. Hiển thị lại các tin nhắn cũ
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 7. Xử lý khi người dùng gửi tin nhắn mới
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Lưu và hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Đợi và hiển thị phản hồi từ AI
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("NemoAI đang suy nghĩ..."):
            try:
                # Chuyển đổi lịch sử chat sang định dạng của Gemini API
                formatted_history = []
                for m in st.session_state.messages[:-1]:
                    role = "user" if m["role"] == "user" else "model"
                    formatted_history.append({"role": role, "parts": [m["content"]]})

                chat = model.start_chat(history=formatted_history)
                response = chat.send_message(prompt)
                ai_response = response.text
                
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                error_msg = f"⚠️ Đã xảy ra lỗi khi kết nối API: {e}"
                st.error(error_msg)
