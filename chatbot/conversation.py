import datetime
import os
import time
from typing import NamedTuple, Optional, Tuple, List
import json


class ChatbotMessage:
    def __init__(self, sender: str, message: str):
        self.sender = sender
        self.message = message
        self.timestamp = time.time()


class Conversation(object):
    def __init__(self, id: str, logdir: Optional[str] = None):
        self.id = id
        self.__queue: list(ChatbotMessage) = []
        self.start_offset = 0
        self.logdir = logdir

    def add_message(self, message: ChatbotMessage):
        self.__queue.append(message)
        self.dump()

    def get_last_message(self, sender: str = None) -> Tuple[int, ChatbotMessage]:
        if sender is None:
            return self.__queue[-1]

        for i in range(len(self.__queue) - 1, -1, -1):
            if self.__queue[i].sender == sender:
                return i, self.__queue[i]

    def dequeue(self):
        self.start_offset += 1

    def get_queue(self) -> List[ChatbotMessage]:
        return self.__queue[self.start_offset :]

    def amend(self, idx: int, message: ChatbotMessage):
        self.__queue[idx] = message

    queue = property(fget=get_queue)

    def dump(self):
        if self.logdir is None:
            return

        if os.path.isfile(self.logdir):
            raise ValueError("logdir is a file")

        if not os.path.isdir(self.logdir):
            os.mkdir(self.logdir)

        with open(os.path.join(self.logdir, f"{self.id}.json"), "w") as f:
            f.write(json.dumps([a.__dict__ for a in self.__queue], indent=4))

    def summary(self, full=False) -> str:
        out = ""
        msgs = self.__queue if full else self.queue

        for message in msgs:
            out += f"{message.sender}: {message.message}\n"

        return out
