from .conversation import *
import os
import re
import codecs
import logging
from typing import Callable

UpdateFunc = Callable[[str], None]

logger = logging.getLogger(__name__)

with open(os.path.join(os.path.dirname(__file__), "slurs-encoded.txt"), "r") as f:
    slurs = f.read().splitlines()
    slurs = map(lambda x: codecs.decode(x, "rot13"), slurs)
    slurs = list(slurs)
    slurs = "|".join(slurs)
    slurs = re.compile(slurs)


def has_slur(message: str):
    return slurs.search(message.lower()) is not None


class Chatbot(object):
    def __init__(self, name: str, preamble: str = "", force_cpu: bool = False, **kwargs):
        self.name = name
        self.force_cpu = force_cpu
        self.preamble = preamble

        self._init_model(**kwargs)

    def model_max_length(self) -> str:
        return "\u221e"

    def _init_model(self, **kwargs):
        pass

    def generate_response(self, convo: Conversation, update: UpdateFunc) -> str:
        _response = ""

        def _update(response: str):
            if has_slur(response):
                logger.info(f"Generated response containing slur: {response}")
                return

            if response != "":
                _response = response
                update(response)

        self._generate(convo, _update)
        convo.add_message(ChatbotMessage(self.name, _response))

        return _response

    def _generate(self, convo: Conversation, update: UpdateFunc):
        pass


class BruhChatbot(Chatbot):
    def _generate(self, convo: Conversation, update: Callable[[str], None]):
        update("bruh")

    def _generate_model_input(self, convo: Conversation) -> str:
        return ""


def test(**kwargs):
    print("============== Chatbot Tester ==============")
    name = input("Enter your name: ")
    for i, subclass in enumerate(Chatbot.__subclasses__()):
        print(f"{i}: {subclass.__name__}")

    choice = -1
    while choice < 0 or choice > len(Chatbot.__subclasses__()):
        choice = int(input("Select a chatbot: "))

    chatbot = Chatbot.__subclasses__()[choice](**kwargs)
    conversation = Conversation("test", logdir="chatlogs")

    print("Loaded Chatbot\n")

    while True:
        try:
            message = input(f"{name}: ")
            conversation.add_message(ChatbotMessage(name, message))

            def update(response: str):
                print(f"\r{chatbot.name}: {response}", end="", flush=True)

            final_response = chatbot.generate_response(conversation, update)
            print("")

        except KeyboardInterrupt:
            break

    print("\n\n============ Summary ============")
    # print(conversation.summary())
    print(chatbot._generate_model_input(conversation))
    # conversation.reset()


if __name__ == "__main__":
    test(name="Bot")
