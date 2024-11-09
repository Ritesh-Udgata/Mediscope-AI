import google.generativeai as genai
GOOGLE_API_KEY='AIzaSyA8x-414brrX8bNB2s_DB8mMGk6wViquJo'
genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

while True:
    prompt = input("Ask me anything: ")
    if (prompt == "exit"):
        break
    response = chat.send_message(prompt, stream=False)
    print(response.text)