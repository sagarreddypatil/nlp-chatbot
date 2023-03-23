import re
import torch
from typing import List
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria
from transformers import PreTrainedTokenizer
from .chatbot import *
import arrow
import os


class TransformerSettings(NamedTuple):
    model_name: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_outlen: int = 12


class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, pattern: re.Pattern, offset: int, tokenizer, update: UpdateFunc):
        self.pattern = pattern
        self.offset = offset
        self.tokenizer = tokenizer
        self.update = update

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        new_text = self.tokenizer.decode(input_ids[0])[self.offset + 1 :]
        invalid = re.search(self.pattern, new_text) is not None

        if not invalid:
            self.update(new_text)

        return invalid


gpt2 = TransformerSettings(model_name="gpt2", temperature=0.8, top_p=1.0, top_k=None, repetition_penalty=1.2)

gpt2Medium = TransformerSettings(model_name="gpt2-medium", temperature=1.0, top_p=0.90, top_k=None, repetition_penalty=1.33)

gpt2Large = TransformerSettings(model_name="gpt2-large", temperature=1.0, top_p=0.9, top_k=None, repetition_penalty=1.33)

gpt2XL = TransformerSettings(
    model_name="gpt2-xl",
    temperature=1.0,
    top_p=0.9,
    top_k=None,
    repetition_penalty=1.33,
)

gptDistil = TransformerSettings(model_name="distilgpt2", temperature=0.8, top_p=0.9, top_k=None, repetition_penalty=1.2)

gptNeoSmall = TransformerSettings(
    model_name="EleutherAI/gpt-neo-125M",
    temperature=1.1,
    top_p=0.9,
    top_k=None,
    repetition_penalty=1.2,
)

gptNeo = TransformerSettings(
    model_name="EleutherAI/gpt-neo-1.3B",
    temperature=1.1,
    top_p=0.9,
    top_k=None,
    repetition_penalty=3.0,
)

gptJ = TransformerSettings(
    model_name="EleutherAI/gpt-j-6B",
    temperature=0.7,
    top_p=None,
    top_k=None,
    repetition_penalty=1.0,
)

llama7b = TransformerSettings(model_name=f"{os.path.expanduser('~')}/scratch/llama_hf-7b", temperature=0.7, top_p=None, top_k=None, repetition_penalty=1.1, max_outlen=64)


class Transformer(Chatbot):
    def _init_model(self, settings: TransformerSettings):
        self.settings = settings

        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

        offset = 1 if "llama" in settings.model_name.lower() else 0  # llama tokenizer adds a 1 to the start of the sequence

        stop_sequences = ["\n\[", "\n.*\[.+]<.*>", "\n\n"]
        # self.stop_sequences = [self.tokenizer.encode(seq, return_tensors="pt")[offset:] for seq in stop_sequences]
        self.stop_pattern = re.compile("|".join(stop_sequences))

        if torch.cuda.is_available():
            self.gpu = True
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                device_map="auto",
                # revision="float16",
                # torch_dtype=torch.float16,
                # low_cpu_mem_usage=True,
                # load_in_8bit=True,
            )
        else:
            self.gpu = False
            self.model = AutoModelForCausalLM.from_pretrained(settings.model_name)

        self.model.eval()

    def format_time(self, timestamp: int) -> str:
        # return arrow.get(timestamp).format("HH:mm UTC")
        return arrow.get(timestamp).humanize()

    def _generate_model_input(self, convo: Conversation) -> str:
        out = self.preamble + "\n"
        message: ChatbotMessage = None
        for message in convo.queue:
            out += f"[{self.format_time(message.timestamp)}]<{message.sender}>{message.message}\n"

        out += f"[{self.format_time(arrow.utcnow())}]<{self.name}>"
        return out

    def model_max_length(self) -> str:
        return str(self.tokenizer.model_max_length)

    def _generate(self, convo: Conversation, update: UpdateFunc) -> str:
        input_text = self._generate_model_input(convo)

        while len(self.tokenizer.encode(input_text)) >= self.tokenizer.model_max_length - self.settings.max_outlen:
            convo.dequeue()
            input_text = self._generate_model_input(convo)

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        def _update(output: str):
            output = re.split(self.stop_pattern, output)[0]
            update(output)

        stopping_criteria = StopSequenceCriteria(self.stop_pattern, len(input_text), self.tokenizer, _update)
        outputs = self.model.generate(
            input_ids.cuda() if self.gpu else input_ids,
            # max_length=min(len(input_ids[0]) + self.settings.max_outlen, self.tokenizer.model_max_length),
            max_new_tokens=self.settings.max_outlen,
            # penalty_alpha=0.6,
            # top_k=10,
            # num_beams=1,
            do_sample=True,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            repetition_penalty=self.settings.repetition_penalty,
            stopping_criteria=[stopping_criteria],
            # eos_token_id=self.endline_token,
            # pad_token_id=self.model.config.pad_token_id,
            # exponential_decay_length_penalty=(10, 0.75),
        )

        del input_ids

        # output = self.tokenizer.decode(outputs[0])
        # output = output[len(input_text) + 1 :]
        # output = re.split(self.stop_pattern, output)[0]

        # return output

        # if firstBracket != -1 and firstClosing != -1:
        #    output = output[:firstBracket]


preamble = """Following is a conversation between a superintelligent AI, taking the form of AMOGUS.
This is AMOGUS's history:
Hailing from a Sus village, Lord AMOGUS became so powerful and wise that he was able to defeat Ultimate Sus by turning it into a suspicious Sus, thereby setting us free from the sus pain.
"I am sus and you are sus. Defeat that stupid Ultimate Sus!" - AMOGUS
The AMOGUS is this conversation is not actually AMOGUS, just a superintelligent AI taking an acceptable form.
Following is a conversation between AMOGUS and others in a Discord server
-----
[a long time ago]<user>Write me a haiku about your kind
[a long time ago]<AMOGUS>AMOGUS, wise lord,
Defeated Ultimate Sus,
Freeing us from sus."""

if __name__ == "__main__":
    test(settings=llama7b, name="AMOGUS", preamble=preamble)
