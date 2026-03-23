import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_audio = "/home/stephen/projects/qwen-tts-installer/outputs/reference.wav"
file_path = "/home/stephen/projects/qwen-tts-installer/outputs/reference.txt"
ref_text = ""
with open(file_path, 'r', encoding='utf-8') as file:
    ref_text = file.read()

script="This is a test of the emergency broadcast system. This is only a test."


"""

do_sample:
    Whether to use sampling, recommended to be set to `true` for most use cases.
top_k:
    Top-k sampling parameter.
top_p:
    Top-p sampling parameter.
temperature:
    Sampling temperature; higher => more random.
repetition_penalty:
    Penalty to reduce repeated tokens/codes.
subtalker_dosample:
    Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
subtalker_top_k:
    Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
subtalker_top_p:
    Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
subtalker_temperature:
    Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
max_new_tokens:
    Maximum number of new codec tokens to generate.
**kwargs:
    Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
    They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

"""

common_gen_kwargs = dict(
    max_new_tokens=2048,
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
    subtalker_dosample=True,
    subtalker_top_k=50,
    subtalker_top_p=1.0,
    subtalker_temperature=0.9,
)

prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False,
)
wavs, sr = model.generate_voice_clone(
    text=[script, script],
    language=["English", "English"],
    voice_clone_prompt=prompt_items,
    **common_gen_kwargs,
)
sf.write("output_voice_clone_1.wav", wavs[0], sr)
sf.write("output_voice_clone_2.wav", wavs[1], sr)
