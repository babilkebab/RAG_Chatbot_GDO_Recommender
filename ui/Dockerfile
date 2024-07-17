FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip install streamlit python-dotenv

COPY ./chat_ui.py ./

ENTRYPOINT ["streamlit", "run", "chat_ui.py"]