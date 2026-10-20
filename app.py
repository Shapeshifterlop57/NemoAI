import streamlit as st
from groq import Groq

st.set_page_config(page_title="NemoAI Pro", page_icon="🤖")

api_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.title("⚙️ Cấu hình NemoAI")
    if not api_key:
        api_key = st.text_input("Nhập Groq API Key (gsk_...):", type="password")
        st.markdown("[👉 Lấy Groq API Key miễn phí](https://console.groq.com/)")

st.title("🤖 NemoAI Pro (Groq Speed Engine)")

if not api_key:
    st.info("💡 Vui lòng nhập Groq API Key ở thanh bên để bắt đầu!")
    st.stop()

client = Groq(api_key=api_key.strip())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI chạy trên hạ tầng Groq siêu tốc. Tôi có thể giúp gì cho bạn?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("NemoAI đang phản hồi..."):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    model="llama3-8b-8192",  # Mô hình Llama 3 cực mượt
                )
                ai_response = chat_completion.choices[0].message.content
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                st.error(f"⚠️ Lỗi kết nối API: {e}")import streamlit as st
from google import genai
from google.genai import types

# 1. Cấu hình giao diện trang Streamlit
st.set_page_config(
    page_title="NemoAI Pro Workspace",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Quản lý API Key / Token
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.title("⚙️ Cấu hình NemoAI")
    
    if not api_key:
        api_key = st.text_input("Nhập Google Cloud Token / API Key:", type="password")
    else:
        st.success("✅ Đã kết nối Key từ Secrets!")

    st.divider()
    
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 NemoAI Pro")
st.caption("Trợ lý AI thông minh, phản hồi thời gian thực trên đám mây.")

if not api_key:
    st.info("💡 Vui lòng nhập khóa ở thanh bên để bắt đầu!")
    st.stop()

# Khởi tạo client hỗ trợ Google Cloud / Vertex AI
try:
    # Nếu dùng token AQ..., ta kích hoạt chế độ Vertex AI với project của bạn
    client = genai.Client(
        vertexai=True,
        project="971262343898",
        location="us-central1"
    )
except Exception as e:
    st.error(f"⚠️ Lỗi khởi tạo Client: {e}")
    st.stop()

# 3. Lịch sử trò chuyện
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI. Tôi có thể giúp gì cho bạn hôm nay?"}
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
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=prompt
                )
                ai_response = response.text
                
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"⚠️ Đã xảy ra lỗi khi kết nối API: {e}")
