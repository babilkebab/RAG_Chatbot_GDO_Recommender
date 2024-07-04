import streamlit as st
import time
import requests



def get_recommendation(question):
    chars_to_replace = ["\\", "&", "%", '"']
    recommendation = requests.get(f"http://localhost:8000/query?question={question}")
    answer = recommendation.text.replace("\\n", "\n")
    for char in chars_to_replace:
        answer = answer.replace(char, "")
    return answer


def typewriting_effect(answer):
    for char in answer:
        yield char
        time.sleep(0.01)


if __name__ == '__main__':
    st.title("Product Recommendation Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])




    prompt = st.chat_input("Scrivi qualcosa...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            recommendation = get_recommendation(prompt)
        with st.chat_message("assistant"):
            st.write_stream(typewriting_effect(recommendation))
            st.session_state.messages.append({"role": "assistant", "content": recommendation})