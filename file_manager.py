import os

def save_queries_to_file(title: str, queries: list):
    """
    Salva os parâmetros title e queries em um arquivo output.txt no formato de texto legível com suporte a UTF-8.
    
    Args:
        title (str): O título do conjunto de queries.
        queries (list): Lista de dicionários contendo 'input' e 'output'.
    """
    file_name = "output.txt"

    # Cria o arquivo se ele não existir
    if not os.path.exists(file_name):
        with open(file_name, 'w', encoding='utf-8') as file:
            pass  # Apenas cria o arquivo vazio

    # Abre o arquivo em modo append para não sobrescrever o conteúdo existente
    with open(file_name, 'a', encoding='utf-8') as file:
        # Adiciona o título
        file.write(f"Título: {title}\n\n")
        
        # Adiciona cada query no formato desejado
        for query in queries:
            input_text = query["input"]
            output_text = query["output"]
            file.write(f"input: {input_text}\n")
            file.write(f"output: {output_text}\n\n")  # Quebra de linha após cada query
        
        # Linha em branco para separar seções no arquivo
        file.write("\n")

    print(f"Dados salvos com sucesso em {file_name}.")
