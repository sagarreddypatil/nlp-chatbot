import gc
import arrow
import os
import re
from dataclasses import dataclass
from dotenv import load_dotenv

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import SinkCache
from transformers import TextIteratorStreamer
from transformers import StoppingCriteria

from .conversation import *
from .formatter import Formatter

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

@dataclass
class ModelSettings:
    model_path: str

    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float

llama7b = ModelSettings(model_path=f"{os.path.expanduser('~')}/scratch/llama_hf-7b", temperature=0.7, top_p=None, top_k=None, repetition_penalty=1.1)
llama13b = ModelSettings(model_path=f"{os.path.expanduser('~')}/scratch/llama_hf-13b", temperature=0.7, top_p=None, top_k=None, repetition_penalty=1.1)
llama27b = ModelSettings(model_path="meta-llama/Llama-2-7b-hf", temperature=0.7, top_p=None, top_k=None, repetition_penalty=1.1)

class Transformer:
    def __init__(self, settings: ModelSettings, formatter: Formatter):
        self.settings = settings
        self.formatter = formatter

        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name, legacy=False, token=HF_TOKEN)
        self.gpu = torch.cuda.is_available()

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            device_map="auto" if self.gpu else "cpu",
            torch_dtype=torch.float16,
            token=HF_TOKEN,
        )
        self.device = self.model.device
        self.model.eval()

        self.cache = SinkCache(window_length=self.tokenizer.model_max_length, num_sink_tokens=4)
        self.stc = RegexStoppingCriteria(self.formatter.stop_pattern, self.tokenizer)

    def generate(self, convo: Conversation) -> str:
        input_text = self.formatter.format(convo)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", device=self.device)

        self.stc.prompt_length(len(input_text))
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        _ = self.model.generate(
            input_ids,
            streamer=streamer,

            do_sample=True,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            repetition_penalty=self.settings.repetition_penalty,
            stopping_criteria=[self.stc],

            use_cache=True,
            past_key_values=self.cache
        )

        del input_ids
        gc.collect()
        torch.cuda.empty_cache()

        output = ""
        for new_text in streamer:
            output += new_text



class RegexStoppingCriteria(StoppingCriteria):
    def __init__(self, pattern: re.Pattern, tokenizer):
        self.pattern = pattern
        self.tokenizer = tokenizer
        self.prompt_length = 0

    def prompt_length(self, new_length: int):
        self.prompt_length = new_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids)
        new_text = text[self.prompt_length:]
        invalid = re.search(self.pattern, new_text) is not None

        return invalid