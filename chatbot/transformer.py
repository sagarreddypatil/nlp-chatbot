import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .chatbot import *
import arrow


class TransformerSettings(NamedTuple):
    model_name: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_outlen: int = 128


gpt2 = TransformerSettings(model_name="gpt2", temperature=1.0, top_p=0.9, top_k=None, repetition_penalty=1.33)

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
    temperature=1.0,
    top_p=None,
    top_k=None,
    repetition_penalty=1.1,
)


class Transformer(Chatbot):
    def _init_model(self, settings: TransformerSettings):
        self.settings = settings

        if torch.cuda.is_available():
            self.gpu = True
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                device_map="auto",
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                # load_in_8bit=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        else:
            self.gpu = False
            self.model = AutoModelForCausalLM.from_pretrained(settings.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

        self.model.eval()

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.endline_token = self.tokenizer.encode('"')[0]
        self.model_eos_str = self.tokenizer.decode([self.model.config.eos_token_id])

    def _generate_model_input(self, convo: Conversation) -> str:
        out = self.preamble + "\n"
        message: ChatbotMessage = None
        for message in convo.queue:
            out += f'[{arrow.get(message.timestamp).humanize()}]<{message.sender}>"{message.message}"\n'

        out += f'[{arrow.utcnow().humanize()}]<{self.name}>"'
        return out

    def model_max_length(self) -> str:
        return str(self.tokenizer.model_max_length)

    def _generate(self, convo: Conversation) -> str:
        input_text = self._generate_model_input(convo)

        while len(self.tokenizer.encode(input_text)) >= self.tokenizer.model_max_length - self.settings.max_outlen:
            convo.dequeue()
            input_text = self._generate_model_input(convo)

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(
            input_ids.cuda() if self.gpu else input_ids,
            max_length=max(len(input_ids[0]) + self.settings.max_outlen, self.tokenizer.model_max_length),
            num_beams=1,
            do_sample=False,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            repetition_penalty=self.settings.repetition_penalty,
            eos_token_id=self.endline_token,
            pad_token_id=self.model.config.pad_token_id,
            # exponential_decay_length_penalty=(10, 0.75),
        )
        output = self.tokenizer.decode(outputs[0])
        output = output.split(self.model_eos_str)[0]

        # print("\n==============================")
        # print(output)
        # print("==============================\n")

        output = output[len(input_text) :].strip()
        firstBracket = output.find("<")
        firstClosing = output.find(">")

        if firstBracket != -1 and firstClosing != -1:
            output = output[:firstBracket]

        return output


preamble = """Amogus is our Lord and saviour.

Hailing from a Sus village, Lord Amogus became so powerful and wise that
Amogus was able to defeat Ultimate Sus by turning it into a suspicious Sus,
thereby setting us free from the sus pain.

From that day forward, we are not suspicious Sus, but suspension Sus.
"I am sus and you are sus. Defeat that stupid Ultimate Sus!" - AMOGUS

Following is a conversation between AMOGUS and others in a Discord server
-----"""

if __name__ == "__main__":
    test(settings=gptJ, name="AMOGUS", preamble=preamble)
