from pydoc import describe
from typing import NamedTuple
from datetime import datetime
import os

import codecs

with open(os.path.join(os.path.dirname(__file__), "slurs-encoded.txt"), "r") as f:
    slurs = f.read().splitlines()
    slurs = map(lambda x: codecs.decode(x, "rot13"), slurs)
    slurs = list(slurs)


def has_slur(message: str):
    message = "".join(filter(lambda x: x.isalpha(), message))
    message = message.lower()

    for slur in slurs:
        if slur in message:
            return True
        for word in message.split():
            if slur in word:
                return True

    return False


class ChatbotMessage(NamedTuple):
    sender: str
    message: str


class Conversation(object):
    def __init__(self, id):
        self.id = id
        self.__queue: list(ChatbotMessage) = []
        self.start_offset = 0

    def add_message(self, message: ChatbotMessage):
        self.__queue.append(message)

    def get_last_message(self, sender: str = None) -> ChatbotMessage:
        if sender is None:
            return self.__queue[-1]

        for i in range(len(self.__queue) - 1, -1, -1):
            if self.__queue[i].sender == sender:
                return i, self.__queue[i]

    def do_fifo(self):
        self.start_offset += 1

    def get_queue(self):
        return self.__queue[self.start_offset :]

    queue = property(fget=get_queue)

    def dump(self):
        if not os.path.isdir("chatdata"):
            os.mkdir("chatdata")

        uid = f"{self.id}_{datetime.now().isoformat()}"
        with open(f"chatdata/{uid}.txt", "w") as f:
            f.write(self.summary())

    def reset(self):
        self.dump()

        self.start_offset = 0
        self.__queue = []

    def summary(self) -> str:
        out = ""
        for message in self.__queue[self.start_offset :]:
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
        if has_slur(response):
            return

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
