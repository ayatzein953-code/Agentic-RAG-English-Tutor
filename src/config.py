import os

# المسارات الأساسية
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(file)))
PDF_PATH = os.path.join(BASE_DIR, "data", "English.pdf")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "chroma_db_english")
LOCAL_MODEL_PATH = r"C:/Users/ayaat/Desktop/ai_model" # أو مسار نموذج التضمين الخاص بكِ

# إعدادات Ollama
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"
