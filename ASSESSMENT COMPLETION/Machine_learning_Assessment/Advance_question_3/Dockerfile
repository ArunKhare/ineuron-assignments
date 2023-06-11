FROM python:3.10

WORKDIR /app

COPY ../requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /app/Advance_question_3

WORKDIR /app/Advance_question_3

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
