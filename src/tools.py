from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from src.config import PERSIST_DIRECTORY, LOCAL_MODEL_PATH, OLLAMA_MODEL, OLLAMA_BASE_URL

# --- تجهيز أداة الـ RAG ---
embeddings = HuggingFaceEmbeddings(
    model_name=LOCAL_MODEL_PATH,
    model_kwargs={'device': 'cpu'}
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY, 
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def search_english_documents(query: str) -> str:
    """
    استخدم هذه الأداة للبحث في مستندات وقواعد تعليم اللغة الإنجليزية المتاحة والمحفوظة محلياً.
    مفيدة جداً عندما يسأل المستخدم عن مفردات، قواعد، أو مواضيع متعلقة بملف الإنجليزية الخاص به.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "لم يتم العثور على معلومات ذات صلة في المستندات."
    return "\n\n".join(doc.page_content for doc in docs)

# --- أداة الترجمة ---
@tool
def translate_text(text: str, target_language: str = "Arabic") -> str:
    """
    استخدم هذه الأداة لترجمة النصوص بين اللغة الإنجليزية واللغة العربية بدقة عالية.
    قم بتمرير النص المراد ترجمته مع اللغة المستهدفة.
    """
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    prompt = f"قم بترجمة النص التالي بدقة مع الحفاظ على المعنى إلى اللغة ({target_language}):\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

# --- أداة تصحيح الأخطاء اللغوية ---
@tool
def check_grammar(text: str) -> str:
    """
    استخدم هذه الأداة لتصحيح الأخطاء النحوية والإملائية في الجمل الإنجليزية وتقييم أسلوب الكتابة.
    تُعيد الجملة المصححة مع شرح سريع ولطيف للأخطاء إن وجدت.
    """
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    prompt = (
        "أنت مصحح لغوي محترف للغة الإنجليزية. قم بتحليل النص التالي:\n"
        f"النص: '{text}'\n\n"
        "قدم الإجابة بالتنسيق التالي:\n"
        "1. النص المصحح (Corrected Version)\n"
        "2. الأخطاء وشرحها المباشر (Explanation of Errors)\n"
        "3. اقتراح تحسين التعبير (Better Formulation) إن أمكن."
    )
    response = llm.invoke(prompt)
    return response.content

# تجميع كل الأدوات في قائمة واحدة سهلة التصدير
all_tools = [search_english_documents, translate_text, check_grammar]
