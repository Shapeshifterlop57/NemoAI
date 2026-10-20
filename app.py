import streamlit as st
from groq import Groq

st.set_page_config(page_title="NemoAI Pro", page_icon="🤖", layout="centered")

api_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.title("⚙️ Cấu hình NemoAI")
    if not api_key:
        api_key = st.text_input("Nhập Groq API Key (gsk_...):", type="password")
    else:
        st.success("✅ Đã kết nối Groq API Key!")
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 NemoAI Pro")
st.caption("Trợ lý AI phản hồi siêu tốc độ.")

if not api_key:
    st.info("💡 Vui lòng nhập Groq API Key ở thanh bên để bắt đầu!")
    st.stop()

try:
    client = Groq(api_key=api_key.strip())
except Exception as e:
    st.error(f"⚠️ Lỗi khởi tạo Groq Client: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI. Tôi có thể giúp gì cho bạn?"}
    ]

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("NemoAI đang suy nghĩ..."):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    model="llama-3.1-8b-instant",  # Model chính thức, ổn định
                )
                ai_response = chat_completion.choices[0].message.content
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                st.error(f"⚠️ Lỗi kết nối API: {e}")
