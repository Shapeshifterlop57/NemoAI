import streamlit as st
from google import genai

# 1. Cấu hình giao diện trang Streamlit
st.set_page_config(
    page_title="NemoAI Pro Workspace",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Quản lý API Key (Tự động lấy từ Secrets hoặc Sidebar)
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.title("⚙️ Cấu hình NemoAI")
    
    if not api_key:
        api_key = st.text_input("Nhập Google Gemini API Key:", type="password")
        st.markdown("[👉 Lấy Gemini API Key miễn phí](https://aistudio.google.com/)")
    else:
        st.success("✅ Đã kết nối API Key từ Secrets!")

    st.divider()
    
    # Nút xóa lịch sử cuộc trò chuyện
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 3. Tiêu đề ứng dụng
st.title("🤖 NemoAI Pro")
st.caption("Trợ lý AI thông minh, phản hồi thời gian thực trên đám mây.")

# 4. Kiểm tra xem đã có API Key chưa
if not api_key:
    st.info("💡 Vui lòng nhập Gemini API Key ở thanh bên (Sidebar) để bắt đầu trò chuyện!")
    st.stop()

# Khởi tạo client Gemini với SDK mới
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"⚠️ Lỗi khởi tạo API Client: {e}")
    st.stop()

# 5. Khởi tạo danh sách lưu trữ lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI. Tôi có thể giúp gì cho bạn hôm nay?"}
    ]

# 6. Hiển thị lại lịch sử các tin nhắn
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 7. Xử lý khi người dùng nhập câu hỏi mới
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Gọi API nhận câu trả lời từ Gemini
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("NemoAI đang suy nghĩ..."):
            try:
                # Gửi yêu cầu qua mô hình gemini-2.5-flash / gemini-1.5-flash
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                ai_response = response.text
                
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                # Nếu gemini-2.5-flash báo lỗi, thử fallback sang gemini-1.5-flash
                try:
                    response = client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=prompt
                    )
                    ai_response = response.text
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                except Exception as ex:
                    st.error(f"⚠️ Đã xảy ra lỗi khi kết nối API: {ex}")
