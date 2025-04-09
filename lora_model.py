from transformers import LlamaForCausalLM, GenerationConfig
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import sys
import torch
from peft import PeftModel
import transformers
import gradio as gr

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"


SHARE_GRADIO = True
LOAD_8BIT = False

BASE_MODEL = "mrzlab630/weights_Llama_7b"
LORA_WEIGHTS = "mrzlab630/lora-alpaca-trading-candles"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

# This is the fixed version of your model loading code

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder=r'./offload_dir',
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder=r'./offload_dir',
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
        offload_folder=r'./offload_dir',
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    # Added offload_folder for CPU case
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        low_cpu_mem_usage=True,
        offload_folder=r'./offload_dir'  # Add this line
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Slider(minimum=0, maximum=1,
                             value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100,
                             step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=4, step=1,
                             value=4, label="Beams"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        ),
    ],
    outputs=[
        gr.components.Textbox(  # Changed from gr.inputs.Textbox
            lines=5,
            label="Output",
        )
    ],
    title="💹 🕯 Alpaca-LoRA-Trading-Candles",
    description="Alpaca-LoRA-Trading-Candles is a 7B-parameter LLaMA model tuned to execute instructions. It is trained on the [trading candles] dataset(https://huggingface.co/datasets/mrzlab630/trading-candles) and uses the Huggingface LLaMA implementation. For more information, visit [project website](https://huggingface.co/mrzlab630/lora-alpaca-trading-candles).\nPrompts:\nInstruction: identify candle, Input: open:241.5,close:232.9, high:241.7, low:230.8\nInstruction: find candle, Input: 38811.24,38838.41,38846.71,38736.24,234.00,45275276.00,59816.00,441285.00,645.00,84176.00,1694619.00,15732335.00\nInstruction: find candle: Bullish, Input: 38751.32,38818.6,38818.6,38695.03,62759348.00,2605789.00,71030.00,820738.00,59659.00,724738.00,7368363.00,50654.00",
).launch(server_name="0.0.0.0", share=SHARE_GRADIO)
