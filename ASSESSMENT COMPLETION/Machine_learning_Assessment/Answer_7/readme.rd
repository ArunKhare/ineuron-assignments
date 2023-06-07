
streamlit run ./Answer_7/MusicGeners.py

docker build -t musicgeners-app:latest .
docker run -p 8501:8501 musicgeners-app 