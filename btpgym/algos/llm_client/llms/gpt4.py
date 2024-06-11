from openai import OpenAI




class LLMGPT4():
    def __init__(self):
        self.client = OpenAI(
            base_url="https://kapkey.chatgptapi.org.cn/v1",            api_key="sk-YX6Kn2rtcXHmz4en9f6c2931Bf194eD892A09e70E7493fA9"
        )

    def request(self,message):
        completion = self.client.chat.completions.create(
          model="gpt-4-turbo-preview",
          # messages=[
          #   {"role": "system", "content": ""},#You are a helpful assistant.
          #   {"role": "user", "content": question}
          # ]
            messages=message
        )

        return completion.choices[0].message.content

    def embedding(self,question):
        embeddings = self.client.embeddings.create(
          model="text-embedding-ada-002",
          input=question
        )

        return embeddings
if __name__ == '__main__':
    # llm = LLMGPT4()
    # answer = llm.request(question="who are you ?")
    # print(answer)

    llm = LLMGPT4()
    answer = llm.embedding(question="who are you,gpt?")
    print(answer)

    messages = [{"role": "system", "content": ""}]
    while True:
        prompt = input("请输入你的问题:")
        messages.append({"role": "user", "content": prompt})
        res_msg = llm.request(messages)
        messages.append({"role": "assistant", "content": res_msg})
        print(res_msg)