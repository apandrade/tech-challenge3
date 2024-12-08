from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from file_manager import save_queries_to_file
from queries import queries

tokenizer = BartTokenizer.from_pretrained("../models-tuned/bart-10per100/tokenizer")
model = BartForConditionalGeneration.from_pretrained('../models-tuned/bart-10per100/model')

# Definir o decoder_start_token_id e pad_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id

# Criar a configuração de geração
generation_config = GenerationConfig(
    decoder_start_token_id=tokenizer.cls_token_id,
    max_length=256,
    num_beams=4,
    no_repeat_ngram_size=2,
    top_k=20,
    top_p=0.0,
    do_sample=True,
    temperature=0.3
)

# Função de tradução
def generate_response(text, max_length=128):
    # Codificar o texto
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)["input_ids"]

    # Gere a tradução com a configuração de geração
    outputs = model.generate(input_ids=input_ids, generation_config=generation_config)

    # Decodifique a saída gerada
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output


title = r"Inferência Bart treinado com 10% da base"
for query in queries:

    input_text = query["input"]
    response = generate_response(input_text)
    query["output"] = response

    print(f"Input: {input_text}")
    print(f"Output: {response}")
    print('\n')


save_queries_to_file(title, queries)
