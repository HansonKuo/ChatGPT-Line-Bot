import json
import datetime


class FileStorage:

  def __init__(self, file_name):
    self.fine_name = file_name
    self.history = {}

  def save(self, data):
    self.history.update(data)
    with open(self.fine_name, 'w', newline='', encoding="utf-8") as f:
      json.dump(self.history, f, ensure_ascii=False)

  def append(self, data):
    self.history.update(data)
    with open(self.fine_name, 'a', newline='', encoding="utf-8") as f:
      json.dump(self.history, f, ensure_ascii=False)

  def load(self):
    with open(self.fine_name, newline='', encoding="utf-8") as jsonfile:
      data = json.load(jsonfile)
    self.history = data
    return self.history


class MongoStorage:

  def __init__(self, db):
    self.db = db

  def save(self, data):
    user_id, api_key = list(data.items())[0]
    self.db['api_key'].update_one({'user_id': user_id}, {
      '$set': {
        'user_id': user_id,
        'api_key': api_key,
        'created_at': datetime.datetime.utcnow()
      }
    },
                                  upsert=True)

  def load(self):
    data = list(self.db['api_key'].find())
    res = {}
    for i in range(len(data)):
      res[data[i]['user_id']] = data[i]['api_key']
    return res


class Storage:

  def __init__(self, storage):
    self.storage = storage

  def save(self, data):
    self.storage.save(data)

  def append(self, data):
    self.storage.append(data)
    
  def load(self):
    return self.storage.load()
