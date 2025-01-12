from prompts import Prompt_for_modify_q, Prompt_for_main_context, Prompt_for_fast_answer, Prompt_for_clarify
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import GigaChat
from openai import OpenAI 
import httpx 

from langchain.vectorstores import Qdrant

from langchain.docstore.document import Document

from telegram import Update
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ApplicationBuilder, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackQueryHandler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def log_message(message: str):
    """Запись сообщения в лог-файл"""
    logging.info(message)

#создание базы данных
def load_txt_file(file_name): 
    with open(file_name, 'r', encoding='utf-8') as file: 
        content = file.read() 
    return Document(page_content=content) 

def cut_chunks(document, chunk_size=1000, overlap_size=200):
    content = document.page_content.replace('\n', ' ').strip()
    chunks = []
    start = 0
    max_length = len(content)

    while start < max_length:
        end = start + chunk_size
        chunk = content[start:end]
        chunks.append(Document(page_content=chunk))
        start += chunk_size - overlap_size  # Шаг с учетом перекрытия

    return chunks

file_name = 'docs.txt' 
document = load_txt_file(file_name) 
documents = cut_chunks(document)

ENCODER_MODEL_NAME = 'intfloat/multilingual-e5-large'
model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceEmbeddings(model_name = ENCODER_MODEL_NAME, model_kwargs=model_kwargs)
location=":memory:"

def load_to_collection(collection_name, documents):
    db = Qdrant.from_documents(
        documents,
        embeddings,
        #location='./',
        location=location,
        collection_name=collection_name,
    )
    return db

db = load_to_collection('info', documents)


#инициализация гигачата
def init_gigachat(temperature, top_p = 0.47):
    gigachat = GigaChat(
        credentials='',
        scope="GIGACHAT_API_CORP",
        model="GigaChat-Pro",
        verify_ssl_certs=False,
        temperature=temperature,
        top_p = top_p,
        #timeout=30
    )

    return gigachat


#запрос в модель
def get_response(system_prompt, user_prompt, temperature = 0.87, history = ''):
  giga = init_gigachat(temperature=temperature)
  messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'assistant', 'content': history},
    {'role': 'user', 'content': user_prompt}
    ]

  answer = giga.invoke(messages)
  return answer.content

def get_response2(system_prompt, user_prompt, history = ''): 
    client = OpenAI(api_key="", 
                    http_client=httpx.Client(proxy="") 
                    ) 
 
    completion = client.chat.completions.create( 
        model="gpt-4o", 
        messages=[ 
            {"role": "system", "content": system_prompt}, 
            {'role': 'assistant', 'content': history},
            {"role": "user", "content": user_prompt} 
        ] 
    ) 
 
    return(completion.choices[0].message.content)

#переформулировка вопроса до корректного и связанного с коровами
def get_correct_query(query):
    system_prompt = Prompt_for_main_context
    user_prompt = Prompt_for_modify_q + f'{query}'
    correct_query = get_response(system_prompt, user_prompt)
    return correct_query

#подтягивание похожего на строку sentence контекста из бд: 5 чанков 
def get_context(sentence, k=5):
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(sentence)
    return docs

#генерация ответа от гигачата без знания контекста из бд
def get_fast_answer(query):
    system_prompt = Prompt_for_main_context
    user_prompt = Prompt_for_fast_answer + f'{query}'
    fast_answer = get_response(system_prompt, user_prompt)
    return fast_answer

#поиск нужных чанков в бд (сравнение с context) -> генерация ответа от гигачата с учетом инфы из бд
def get_answer(query, context):
    system_prompt = Prompt_for_main_context
    user_prompt = query
    docs = get_context(context, k=5)
    answer = get_response2(system_prompt, user_prompt, history = str(docs))
    return answer
   
#обновление вопроса
def get_update_query(query, history):
    Prompt = Prompt_for_modify_q
    answer = get_response2(Prompt, query, history = str(history))
    return answer

#уточнение вопроса (генерация уточняющих вопросов или 'clean')
def get_clarify(query):
    system_prompt = Prompt_for_main_context + Prompt_for_clarify 
    user_prompt = query
    #docs = get_context(query, k=5)
    clarify = get_response2(system_prompt, user_prompt)
    return clarify



START, QUESTION = range(2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:  
    await update.message.reply_text("Привет! Задайте мне вопрос о трудовом праве.")  
    context.user_data['history'] = [] 
    context.user_data['number_of_clarify'] = 0
    log_message("Bot: Привет! Задайте мне вопрос о крс")
    return QUESTION 

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:  
    query = update.message.text  
    log_message(f"user: {query}")
    log_message('новое сообщение от пользователя query = update.message.text')
    log_message(query)
    log_message('--------------------------------')
    
    context.user_data['history'].append({"user": query}) 
    update_query = get_update_query(query, context.user_data['history'])
    log_message('вопрос после подтягивания истории  update_query = get_update_query')
    log_message(update_query)
    log_message(f"вопрос после подтягивания истории: {update_query}")

    if context.user_data['number_of_clarify'] < 1:
        clarify = get_clarify(update_query)
        log_message(f"уточняющие вопросы: {clarify}")
    else:
        clarify = 'clean'

    if clarify != 'clean': 
        log_message('генерирует уточняющие вопросы') 
        log_message(clarify) 
 
        context.user_data['history'].append({"bot": clarify}) 
        context.user_data['number_of_clarify'] += 1
        await update.message.reply_text(clarify)  

    else: 
        await update.message.reply_text('вопрос принят')  
        fast_answer = get_fast_answer(update_query)
        log_message('быстрый ответ')
        log_message(fast_answer)
        log_message(f"быстрый ответ: {fast_answer}")
        log_message('----------------------------------------------------------------')
        answer_from_fast = get_answer(update_query, fast_answer)
        log_message('ответ после быстрого ответа')
        log_message(answer_from_fast)
        log_message(f"ответ после быстрого ответа: {answer_from_fast}")
        log_message('--------------------------------------------------------------')

        doc = get_context(update_query, k=3)
        log_message('контекст для update_query, чтобы получить ответ')
        log_message(doc)
        log_message(f"контекст для update_query, чтобы получить ответ: {doc}")
        log_message('-------------------------------------------------------')
        answer = get_response(Prompt_for_main_context, update_query, history = str(doc))
        log_message('ответ')
        log_message(answer)
        log_message(f"ответ после контекста: {answer}")
        log_message('--------------------------------------------------------------')

        full_answer = get_response(Prompt_for_main_context, update_query, history = answer + answer_from_fast)
        log_message('общий ответ')
        log_message(full_answer)
        log_message(f"общий ответ: {answer}")
        log_message('-----------------------------------------------------------')

        context.user_data['history'].append({"bot": answer})
        await update.message.reply_text(answer)

    keyboard = [ [InlineKeyboardButton("Задать новый вопрос", callback_data='new_question')] ] 
    reply_markup = InlineKeyboardMarkup(keyboard) 

    await update.message.reply_text("Если хотите задать новый вопрос, нажмите кнопку ниже:", reply_markup=reply_markup)
    return QUESTION 
    
async def new_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Очищает историю переписки с ботом 
    context.user_data['history'] = []  
    context.user_data['number_of_clarify'] = 0
    log_message(f"Очищается история переписки с ботом ")

    await context.bot.send_message(chat_id=update.callback_query.from_user.id, text="задайте новый вопрос:") 
    return QUESTION 
    
def main(): 
    app = ApplicationBuilder().token("").build() 
    
    conv_handler = ConversationHandler( 
            entry_points=[CommandHandler('start', start)],
            states={
                QUESTION: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
                    CallbackQueryHandler(new_question, pattern='new_question')
                ]
            },
            fallbacks=[],
            allow_reentry=True,
        ) 

    app.add_handler(conv_handler)
    app.run_polling()


if __name__ == '__main__':
    main()
