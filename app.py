from src.agent import build_agent
from langchain_core.messages import HumanMessage, AIMessage

def main():
    print("🤖 مرحباً بك في ZYNE English Agent - المساعد الذكي لتعلم الإنجليزية!")
    print("يمكنني البحث في ملفاتك، ترجمة النصوص، وتصحيح أخطائك اللغوية.")
    print("اكتب 'exit' للخروج.\n" + "-"*50)

    agent_executor = build_agent()
    chat_history = []

    while True:
        user_input = input("\n👤 أنت: ")
        if user_input.lower() in ["exit", "quit", "خروج"]:
            print("👋 وداعاً! أتمنى لك يوماً سعيداً في التعلم.")
            break

        if not user_input.strip():
            continue

        # استدعاء الـ Agent وتمرير الذاكرة
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        output_text = response["output"]
        print(f"\n🤖 المساعد: {output_text}")

        # تحديث ذاكرة الجلسة الحالية
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output_text))

if name == "main":
    main()
