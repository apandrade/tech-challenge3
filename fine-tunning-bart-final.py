import os
import multiprocessing
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, BartTokenizer, BartForConditionalGeneration
import torch
import numpy as np
import gc

# Hardware usado no processo de treinamento: Xeon E5 2640V3, 16gb de ram e RTX 3060 12gb de vram

model_name = 'facebook/bart-base' 
tokenizer = BartTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = examples["title"]
    targets = examples["content"]
    max_length = 512  # Ajuste conforme necessário
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length)
    labels = tokenizer(text_target=targets, padding="max_length", truncation=True, max_length=max_length)["input_ids"]
    labels = np.array(labels)
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels.tolist()
    return model_inputs


def save_checkpoint(trainer, output_dir="./models-tuned/bart-final"):
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Checkpoint salvo: {output_dir}")


if __name__ == '__main__':
    file_path = './data/trn_sanitized_final.json'
    all_dataset = load_dataset('json', data_files=file_path)
    shuffled_dataset = all_dataset['train'].shuffle(seed=42)

    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsando o dispositivo: {device}\n")
    model.to(device)

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.vocab_size = len(tokenizer)

    print("Iniciando o mapeamento do dataset...")
    tokenized_datasets = shuffled_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,          # Ajuste para fazer melhor uso dos núcleos do processador sem comprometer a estabilidade
        batch_size=100       # Ajuste do tamanho do lote para geração do map
    )
    print("Mapeamento concluído.")

    # Limpar variáveis desnecessárias para liberar memória
    del all_dataset, shuffled_dataset
    gc.collect()

    print("Garbage collected!")
    
    # Configurações de treinamento
    training_args = TrainingArguments(
        output_dir="./data/results",             # Diretório de saída para resultados e checkpoints
        save_steps=1000,                         # Salva os checkpoints a cada 1000 passos
        save_total_limit=2,                      # Manutenção de apenas 2 checkpoints mais recentes
        learning_rate=5e-5,                      # Taxa de aprendizado inicial indicado para 3 epochs
        per_device_train_batch_size=14,          # Tamanho do lote por dispositivo (Tamanho ideal para os 12gb de vram da RTX3060)
        per_device_eval_batch_size=14,           # Tamanho do lote para avaliação (Tamanho ideal para os 12gb de vram da RTX3060)
        num_train_epochs=3,                      # Número de épocas de treinamento
        weight_decay=0.01,                       # Regularização de decaimento de peso
        logging_dir='./logs',                    # Diretório para salvar os logs
        logging_steps=50,                        # Registrar logs a cada 50 passos
        fp16=True,                               # Habilitado meia precisão para economizarmos vram na GPU
        gradient_accumulation_steps=4,           # Acumulação de gradientes para simular um batch_size maior
        seed=42,                                 # Definindo semente para reprodutibilidade
        dataloader_num_workers=8,                # Para fazer melhor uso do multi-thread do processador
        dataloader_pin_memory=False,             # Pode ser ajustado conforme necessário
    )
    
    # Inicializar o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    
    # Treinamento com salvamento ao final
    print("Iniciando o treinamento...")
    
    try:
        trainer.train()
    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento: {e}")

    save_checkpoint(trainer)
        
    print("Treinamento concluído!")
