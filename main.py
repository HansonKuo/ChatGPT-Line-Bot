from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage,
                            ImageSendMessage, AudioMessage)
import os
import uuid
import openai
import pandas as pd
import numpy as np
import pickle

from src.models import OpenAIModel
from src.memory import Memory
from src.logger import logger
from src.storage import Storage, FileStorage, MongoStorage
from src.utils import get_role_and_content
from src.service.youtube import Youtube, YoutubeTranscriptReader
from src.service.website import Website, WebsiteReader
from src.mongodb import mongodb

load_dotenv('.env')

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
storage = None
message_storage = None
youtube = Youtube(step=4)
website = Website()
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"

memory = Memory(system_message=os.getenv('SYSTEM_MESSAGE'),
                memory_message_count=2)
model_management = {}
api_keys = {}

df = pd.read_pickle("embedding_ch.pkl")
document_embeddings = df['embedding']

COMPLETIONS_API_PARAMS = {
  # We use temperature of 0.0 because it gives the most predictable, factual answer.
  "temperature": 0.7,
  "max_tokens": 2048,
  "model": "gpt-3.5-turbo-0301",
}


def get_embedding(text: str, model: str = EMBEDDING_MODEL):
  result = openai.Embedding.create(model=model, input=text)
  return result["data"][0]["embedding"]


def vector_similarity(x, y) -> float:
  """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
  return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts):
  """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
  query_embedding = get_embedding(query)

  document_similarities = sorted(
    [(vector_similarity(query_embedding, doc_embedding), doc_index)
     for doc_index, doc_embedding in contexts.items()],
    reverse=True)

  return document_similarities


def construct_prompt(question: str, context_embeddings: dict,
                     df: pd.DataFrame) -> str:
  """
    Fetch relevant 
    """
  MAX_SECTION_LEN = 1024
  SEPARATOR = "\n* "
  separator_len = 3

  most_relevant_document_sections = order_document_sections_by_query_similarity(
    question, context_embeddings)

  chosen_sections = []
  chosen_sections_len = 0
  chosen_sections_indexes = []

  for _, section_index in most_relevant_document_sections:
    # Add contexts until we run out of space.
    document_section = df.loc[section_index]

    chosen_sections_len += document_section.tokens + separator_len
    if chosen_sections_len > MAX_SECTION_LEN:
      break

    chosen_sections.append(SEPARATOR +
                           document_section.content.replace("\n", " "))
    chosen_sections_indexes.append(str(section_index))

  without_question_mark = question.replace("?", "").replace("？", "").replace(
    "Q:", "").replace("A:", "").replace("\n", "")

  header = f'''You are an occult teacher with compassion and knowledge. When the user is not asking question, you respond concise and short. You are always caring for the user. If the user wasn't asking question, say something interesting based on the context. Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the context, say “答案不確定，我們將加以改進”，,and give a detail summary related to the context and the key word in "{without_question_mark}".Provide more detail based on the context when the user ask to continue.
  \n\nContext:\n
  '''

  return header + "".join(chosen_sections) + "\n\n" + question + "\n A:"


def answer_query_with_context(query: str,
                              df: pd.DataFrame,
                              document_embeddings,
                              show_prompt: bool = False) -> str:
  prompt = construct_prompt(query, document_embeddings, df)

  if show_prompt:
    print(prompt)

  response = openai.ChatCompletion.create(messages=[
    {
      "role": "user",
      "content": prompt
    },
  ],
                                          **COMPLETIONS_API_PARAMS)

  return response["choices"][0]["message"]["content"]  #.strip(" \n")


@app.route("/callback", methods=['POST'])
def callback():
  signature = request.headers['X-Line-Signature']
  body = request.get_data(as_text=True)
  app.logger.info("Request body: " + body)
  try:
    handler.handle(body, signature)
  except InvalidSignatureError:
    print(
      "Invalid signature. Please check your channel access token/channel secret."
    )
    abort(400)
  return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
  user_id = event.source.user_id
  text = event.message.text.strip()
  logger.info(f'{user_id}: {text}')

  try:
    memory.append(user_id, 'Q:', text)
    message_storage.append({user_id: {'user': text}})
    memory_list = memory.get(user_id)
    question = ""
    for item in memory_list:
      answer = item['content'].replace('\n', '')
      question += f"\n {item['role']}{answer}"
    response = answer_query_with_context(question,
                                         df,
                                         document_embeddings,
                                         show_prompt=True)
    print(response)
    message_storage.append({user_id: {'ai': response}})
    memory.append(user_id, 'A:', response[:40])
    print(memory.get(user_id))
    msg = TextSendMessage(text=response)
    line_bot_api.reply_message(event.reply_token, msg)

  except ValueError:
    msg = TextSendMessage(text='Token 無效，請重新註冊，格式為 /註冊 sk-xxxxx')
  except KeyError:
    msg = TextSendMessage(text='請先註冊 Token，格式為 /註冊 sk-xxxxx')
  except Exception as e:
    msg = TextSendMessage(text=str(e))
    print(msg)
  #line_bot_api.reply_message(event.reply_token, msg)


@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
  user_id = event.source.user_id
  audio_content = line_bot_api.get_message_content(event.message.id)
  input_audio_path = f'{str(uuid.uuid4())}.m4a'
  with open(input_audio_path, 'wb') as fd:
    for chunk in audio_content.iter_content():
      fd.write(chunk)

  try:
    if not model_management.get(user_id):
      raise ValueError('Invalid API token')
    else:
      is_successful, response, error_message = model_management[
        user_id].audio_transcriptions(input_audio_path, 'whisper-1')
      if not is_successful:
        raise Exception(error_message)
      memory.append(user_id, 'user', response['text'])
      is_successful, response, error_message = model_management[
        user_id].chat_completions(memory.get(user_id), 'gpt-3.5-turbo')
      if not is_successful:
        raise Exception(error_message)
      role, response = get_role_and_content(response)
      memory.append(user_id, role, response)
      msg = TextSendMessage(text=response)
  except ValueError:
    msg = TextSendMessage(text='請先註冊你的 API Token，格式為 /註冊 [API TOKEN]')
  except KeyError:
    msg = TextSendMessage(text='請先註冊 Token，格式為 /註冊 sk-xxxxx')
  except Exception as e:
    memory.remove(user_id)
    if str(e).startswith('Incorrect API key provided'):
      msg = TextSendMessage(text='OpenAI API Token 有誤，請重新註冊。')
    else:
      msg = TextSendMessage(text=str(e))
  os.remove(input_audio_path)
  line_bot_api.reply_message(event.reply_token, msg)


@app.route("/", methods=['GET'])
def home():
  return 'Hello World'


if __name__ == "__main__":
  if os.getenv('USE_MONGO'):
    mongodb.connect_to_database()
    storage = Storage(MongoStorage(mongodb.db))
  else:
    storage = Storage(FileStorage('db.json'))
    message_storage = Storage(FileStorage('message.json'))
  try:
    data = storage.load()
    # for user_id in data.keys():
    #   model_management[user_id] = OpenAIModel(api_key=data[user_id])
  except FileNotFoundError:
    pass
  app.run(host='0.0.0.0', port=8080)
