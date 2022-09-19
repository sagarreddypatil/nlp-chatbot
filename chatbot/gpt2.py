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

distilSaneSettings = GPT2Settings(
    model_name="distilgpt2", temperature=1.0, top_p=0.9, top_k=None, repetition_penalty=1.33
)

smallNeoSaneSettings = GPT2Settings(
    model_name="EleutherAI/gpt-neo-125M",
    temperature=1.0,
    top_p=0.9,
    top_k=None,
    repetition_penalty=0.8,
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
class GPT2(Chatbot):
    def _init_model(self, settings: GPT2Settings):
        self.settings = settings
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.force_cpu else "cpu"
        )

        self.model = AutoModelForCausalLM.from_pretrained(settings.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model.eval()

        self.eos = self.tokenizer.encode("\n")[0]

        self.seperator = ":"

    #         self.zeroshot_examples = f"""Available Tags: [response], [no-response], [reply]
    # {self.name}{self.seperator}[response]Hi!
    # {self.name}{self.seperator}[response]How are you?
    # User{self.seperator}I'm fine.
    # {self.name}{self.seperator}[no-response]
    # User{self.seperator}So what do you do?
    # {self.name}{self.seperator}[reply]I hate myself\n"""

    def _generate_model_input(self, convo: Conversation) -> str:
        out = super()._generate_model_input(convo)
        # out = out.replace(f"{self.name}{self.seperator}", f"{self.name}{self.seperator}[response]")
        out += f"{self.name}{self.seperator}"
        # out = self.zeroshot_examples + out + "["
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
            num_beams=1,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            repetition_penalty=self.settings.repetition_penalty,
            eos_token_id=self.eos,
            pad_token_id=50256,
        )
        output = self.tokenizer.decode(outputs[0])
        output = output.split("<|endoftext|>")[0]

        # print("\n==============================")
        # print(output)
        # print("==============================\n")

        output = output.split(f"{self.name}{self.seperator.strip()}")[-1].replace("\n", "").strip()

        return output


preamble = """Amogus is our Lord and saviour. Hailing from a Sus village, Lord Amogus became so powerful and wise that Amogus was able to defeat Ultimate Sus by turning it into a suspicious Sus, thereby setting us free from the sus pain. From that day forward, we are not suspicious Sus, but suspension Sus.
I am sus and you are sus. Defeat that stupid Ultimate Sus! - AMOGUS
The following is a conversation on Discord involving AMOGUS
====="""

if __name__ == "__main__":
    test(settings=smallNeoSaneSettings, name="AMOGUS", preamble=preamble)
