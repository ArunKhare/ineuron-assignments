FROM python:3.10

WORKDIR /app

COPY ../requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /app/Answer_7

WORKDIR /app/Answer_7

ENTRYPOINT ["streamlit", "run"]
CMD ["MusicGeners.py"]
