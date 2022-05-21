from pydoc import describe
from typing import NamedTuple
from datetime import datetime
import os


class ChatbotMessage(NamedTuple):
    sender: str
    message: str


class Conversation(object):
    def __init__(self, id):
        self.id = id
        self.queue: list(ChatbotMessage) = []
        self.start_offset = 0

    def add_message(self, message: ChatbotMessage):
        self.queue.append(message)

    def get_last_message(self, sender: str = None) -> ChatbotMessage:
        if sender is None:
            return self.queue[-1]

        for i in range(len(self.queue) - 1, -1, -1):
            if self.queue[i].sender == sender:
                return i, self.queue[i]

    def do_fifo(self):
        self.start_offset += 1

    def reset(self):
        if not os.path.isdir("chatdata"):
            os.mkdir("chatdata")

        uid = f"{self.id}_{datetime.now().isoformat()}"
        with open(f"chatdata/{uid}.txt", "w") as f:
            f.write(self.summary())

        self.start_offset = 0
        self.queue = []

    def summary(self) -> str:
        out = ""
        for message in self.queue[self.start_offset :]:
            out += f"{message.sender}: {message.message}\n"

        return out


class Chatbot(object):
    def __init__(self, name: str, description: str, force_cpu: bool = False, **kwargs):
        self.name = name
        self.description = description
        self.force_cpu = force_cpu
        self.seperator = ": "

        self._init_model(**kwargs)

    def model_max_length(self) -> str:
        return "\u221e"

    def init_conversation(self, convo: Conversation):
        convo.add_message(ChatbotMessage(self.name, "Hello!"))
        for line in self.description.split("\n"):
            convo.add_message(ChatbotMessage(self.name, line))

    def _init_model(self, **kwargs):
        pass

    def generate_response(self, convo: Conversation) -> str:
        response = self._generate(convo)
        convo.add_message(ChatbotMessage(self.name, response))

        return response

    def _generate_model_input(self, convo: Conversation) -> str:
        out = ""
        for message in convo.queue:
            out += f"{message.sender}{self.seperator}{message.message}\n"

        return out

    def _generate(self) -> str:
        pass


class BruhChatbot(Chatbot):
    def _generate(self, convo: Conversation) -> str:
        return "Bruh"


def test(**kwargs):
    print("============== Chatbot Tester ==============")
    for i, subclass in enumerate(Chatbot.__subclasses__()):
        print(f"{i}: {subclass.__name__}")

    choice = -1
    while choice < 0 or choice > len(Chatbot.__subclasses__()):
        choice = int(input("Select a chatbot: "))

    chatbot = Chatbot.__subclasses__()[choice](
        name="Chatbot", description="""I am a chatbot.""", **kwargs
    )
    conversation = Conversation("test")
    chatbot.init_conversation(conversation)

    print("Loaded Chatbot\n")
    name = input("Enter your name: ")

    while True:
        try:
            message = input(f"{name}: ")
            conversation.add_message(ChatbotMessage(name, message))
            response = chatbot.generate_response(conversation)
            print(f"{chatbot.name}: {response}")
        except KeyboardInterrupt:
            break

    print("\n\n============ Summary ============")
    # print(conversation.summary())
    print(chatbot._generate_model_input(conversation))
    conversation.reset()


if __name__ == "__main__":
    test()
