import streamlit as st
import time
import requests
import os
import dotenv

dotenv.load_dotenv()
PORT = os.environ.get("API_PORT")
NUM_OF_EXAMPLES = os.environ.get("NUM_OF_EXAMPLES")


def set_chain_settings(k, model):
    confirm = requests.get(f"http://10.0.100.3:{PORT}/chain_settings?k={k}&model={model}")
    return confirm



def get_recommendation(question):
    chars_to_replace = ["\\", "&", "%", '"']
    recommendation = requests.get(f"http://10.0.100.3:{PORT}/query?question={question}")
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

    if 'submitted' not in st.session_state:
        st.session_state["submitted"] = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state["submitted"]:
        with st.form("my_form"):
            k = st.number_input("Inserisci il valore di k", min_value=0, max_value=int(NUM_OF_EXAMPLES), format="%d")
            model_option = st.selectbox(
                "Quale modello vorresti utilizzare per generare la query?",
                ("GPT 3.5 Turbo", "GPT 4 Turbo", "GPT 4o"),
            )
            submitted = st.form_submit_button("Conferma")
            if submitted:
                confirm = set_chain_settings(k, model_option)
                st.markdown(confirm.text)
                st.session_state["submitted"] = True

    with st.spinner("Generazione della risposta..."):
        if st.session_state["submitted"]:
            prompt = st.chat_input("Scrivi qualcosa...")
            if prompt:
                with st.chat_message("user"):
                    st.markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    recommendation = get_recommendation(prompt)
                with st.chat_message("assistant"):
                    st.write_stream(typewriting_effect(recommendation))
                    st.session_state.messages.append({"role": "assistant", "content": recommendation})