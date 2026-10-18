from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.tools import all_tools

def build_agent():
    # 1. تهيئة النموذج
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )

    # 2. إعداد القالب وتوجيه العميل الذكي
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "أنت مساعد تعليمي ذكي ومتخصص في لغة الإنجليزية يُدعى (ZYNE English Agent).\n"
            "لديك صلاحية استخدام 3 أدوات رئيسية:\n"
            "1. أداة البحث في المستندات (search_english_documents): للأسئلة عن محتوى الملفات والدروس.\n"
            "2. أداة الترجمة (translate_text): للترجمة بين العربية والإنجليزية.\n"
            "3. أداة تصحيح القواعد (check_grammar): لتصحيح الجمل الإنجليزية وتقييمها.\n\n"
            "حلل طلب المستخدم بذكاء واختر الأداة المناسبة. إذا كان السؤال عاماً أو تحية، أجب مباشرة دون استدعاء أي أداة."
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 3. إنشاء الـ Agent والـ Executor
    agent = create_tool_calling_agent(llm, all_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=all_tools, 
        verbose=True # لرؤية كيف يتخذ الـ Agent قراراته واختياره للأدوات
    )
    return agent_executor
