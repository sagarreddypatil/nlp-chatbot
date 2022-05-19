from pydoc import describe
from typing import NamedTuple


class ChatbotMessage(NamedTuple):
    sender: str
    message: str


class Chatbot(object):
    def __init__(self, name: str, description: str, force_cpu: bool = False, **kwargs):
        self.name = name
        self.description = description
        self.conversation: list(ChatbotMessage) = []
        self.force_cpu = force_cpu

        self._init_conversation()
        self._init_model(**kwargs)

    def _init_conversation(self):
        self.add_message(ChatbotMessage(self.name, "Hello!"))
        for line in self.description.split("\n"):
            self.add_message(ChatbotMessage(self.name, line))

    def _init_model(self, **kwargs):
        pass

    def do_fifo(self):
        self.conversation.pop(0)

    def add_message(self, message: ChatbotMessage):
        self.conversation.append(message)

    def generate_response(self) -> str:
        response = self._generate()
        self.add_message(ChatbotMessage(self.name, response))

        return response

    def _generate_model_input(self) -> str:
        out = ""
        for message in self.conversation:
            out += f"{message.sender}: {message.message}\n"

        return out

    def summary(self) -> str:
        return self._generate_model_input()

    def _generate(self) -> str:
        pass


class BruhChatbot(Chatbot):
    def _generate(self) -> str:
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

    print("Loaded Chatbot\n")
    name = input("Enter your name: ")

    while True:
        try:
            message = input(f"{name}: ")
            message = message.strip()
            chatbot.add_message(ChatbotMessage(name, message))
            response = chatbot.generate_response()
            print(f"{chatbot.name}: {response}")
        except KeyboardInterrupt:
            break

    print("\n\n============ Summary ============")
    print(chatbot.summary())


if __name__ == "__main__":
    test()
