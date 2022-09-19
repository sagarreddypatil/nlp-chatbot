import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .chatbot import *


class GPT2Settings(NamedTuple):
    model_name: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_outlen: int = 128


distilInsaneSettings = GPT2Settings(
    model_name="distilgpt2", temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=2.0
)

largeSaneSettings = GPT2Settings(
    model_name="gpt2-large", temperature=1.0, top_p=0.9, top_k=None, repetition_penalty=1.33
)

smallNeoSaneSettings = GPT2Settings(
    model_name="EleutherAI/gpt-neo-125M",
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.33,
)

distilSaneSettings = GPT2Settings(
    model_name="distilgpt2", temperature=0.9, top_p=0.9, top_k=50, repetition_penalty=1.33
)

bigBrainSettings = GPT2Settings(
    model_name="gpt2-xl",
    temperature=1.0,
    top_p=0.9,
    top_k=None,
    repetition_penalty=1.33,
)

neoSettings = GPT2Settings(
    model_name="EleutherAI/gpt-neo-1.3B",
    temperature=0.9,
    top_p=None,
    top_k=None,
    repetition_penalty=50.0,
)


# Sike it's actually distilgpt2
class Transformer(Chatbot):
    def _init_model(self, settings: GPT2Settings):
        self.settings = settings
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.force_cpu else "cpu"
        )

        self.model = AutoModelForCausalLM.from_pretrained(settings.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model.eval()

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.endline_token = self.tokenizer.encode("\n")[0]
        self.model_eos_str = self.tokenizer.decode([self.model.config.eos_token_id])

        self.seperator = ":"

    def _generate_model_input(self, convo: Conversation) -> str:
        out = super()._generate_model_input(convo)
        out += f"{self.name}{self.seperator}"
        return out

    def model_max_length(self) -> str:
        return str(self.tokenizer.model_max_length)

    def _generate(self, convo: Conversation) -> str:
        input_text = self._generate_model_input(convo)

        while (
            len(self.tokenizer.encode(input_text))
            >= self.tokenizer.model_max_length - self.settings.max_outlen
        ):
            convo.dequeue()
            input_text = self._generate_model_input(convo)

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(
            input_ids.to(self.device),
            max_length=max(
                len(input_ids[0]) + self.settings.max_outlen, self.tokenizer.model_max_length
            ),
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            repetition_penalty=self.settings.repetition_penalty,
            eos_token_id=self.endline_token,
            num_beams=1,
            exponential_decay_length_penalty=(20, 0.9),
            do_sample=True,
        )
        output = self.tokenizer.decode(outputs[0])
        output = output.split(self.model_eos_str)[0]

        # print("\n==============================")
        # print(output)
        # print("==============================\n")

        output = output.split(f"{self.name}{self.seperator.strip()}")[-1].replace("\n", "").strip()

        return output


preamble = """Amogus is our Lord and saviour. Hailing from a Sus village, Lord Amogus became so powerful and wise that Amogus was able to defeat Ultimate Sus by turning it into a suspicious Sus, thereby setting us free from the sus pain. From that day forward, we are not suspicious Sus, but suspension Sus.
I am sus and you are sus. Defeat that stupid Ultimate Sus! - AMOGUS
The following is a conversation on Discord involving AMOGUS
-----"""

if __name__ == "__main__":
    test(settings=largeSaneSettings, name="AMOGUS", preamble=preamble)
