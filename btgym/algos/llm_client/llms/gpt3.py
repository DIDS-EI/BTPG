import os
# os.environ["OPENAI_API_KEY"]="sk-4vD6bVtv67XcfoVS8802AdF75888473296D604D707FbC9Bf"
# os.environ["OPENAI_BASE_URL"]= "https://gtapi.xiaoerchaoren.com:8932"

from openai import OpenAI



class LLMGPT3():
    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.xty.app/v1", api_key="sk-FLyhhGWDsCZCTbmq640c5c61Ad3d45078eDe56CdDbF01c0a"
            # base_url="https://gtapi.xiaoerchaoren.com:8932/v1",            api_key="sk-OO5BXh9SUMrnWR6q6fC035142aC94352A59f78E8655fE62b"
        )
    def request(self,message): # question
        completion = self.client.chat.completions.create(
          model="gpt-3.5-turbo",
          # messages=[
          #   {"role": "system", "content": ""},#You are a helpful assistant.
          #   {"role": "user", "content": question}
          # ]
            messages=message
        )

        return completion.choices[0].message.content

    def embedding(self,question):
        embeddings = self.client.embeddings.create(
          model="text-embedding-3-small",
          # model="text-embedding-ada-002",
          input=question
        )

        return embeddings
    def list_models(self):
        response = self.client.models.list()
        return response.data
    def list_embedding_models(self):
        models = self.list_models()
        embedding_models = [model.id for model in models if "embedding" in model.id]
        return embedding_models


if __name__ == '__main__':
    llm = LLMGPT3()
    # embedding_models = llm.list_embedding_models()
    # print("Available embedding models:")
    # for model in embedding_models:
    #     print(model)

    # models = llm.list_models()
    # for model in models:
    #     print(model.id)

    # answer = llm.request(question="who are you,gpt?")
    answer = llm.embedding(question="who are you,gpt?")
    print(answer)

    llm = LLMGPT3()
    messages = [{"role": "system", "content": "你现在是很有用的助手！"}]
    while True:
        prompt = input("请输入你的问题:")
        messages.append({"role": "user", "content": prompt})
        res_msg = llm.request(messages)
        messages.append({"role": "assistant", "content": res_msg})
        print(res_msg)