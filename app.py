import streamlit as st
from transformers import pipeline

# 1. Cấu hình giao diện
st.set_page_config(
    page_title="NemoAI Free", 
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 NemoAI (Không cần API Key)")
st.caption("Chạy hoàn toàn bằng mô hình nguồn mở tích hợp sẵn, không lo lỗi kết nối!")

# 2. Tải mô hình AI trực tiếp vào bộ nhớ (Chỉ tải 1 lần duy nhất)
@st.cache_resource
def load_ai_model():
    # Sử dụng mô hình nhẹ, chạy tốt trên CPU miễn phí của Streamlit
    return pipeline("text2text-generation", model="google/flan-t5-small")

with st.spinner("Đang khởi động hệ thống AI (mất khoảng vài giây)..."):
    try:
        chatbot = load_ai_model()
    except Exception as e:
        st.error(f"⚠️ Không thể khởi động mô hình: {e}")
        st.stop()

# 3. Lịch sử trò chuyện
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là NemoAI phiên bản độc lập. Bạn có thể trò chuyện thoải mái mà không cần nhập bất kỳ API Key nào!"}
    ]

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 4. Xử lý câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("NemoAI đang trả lời..."):
            try:
                # Tạo câu trả lời từ mô hình local
                response = chatbot(prompt, max_length=200, do_sample=True, temperature=0.7)
                ai_response = response[0]['generated_text']
                
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                st.error(f"⚠️ Đã xảy ra lỗi: {e}")
