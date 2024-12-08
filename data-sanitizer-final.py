import json

# Carregar o arquivo original
file_path = "./data/trn.json"
with open(file_path, "r") as file:
    data = [json.loads(line) for line in file]

# Filtrar as colunas "title" e "content"
filtered_data = [{"title": item["title"], "content": item["content"]} for item in data]

# Contagem de vazios, nulos e duplicados
empty_titles = 0
null_titles = 0
empty_contents = 0
null_contents = 0
duplicate_contents = 0

# Rastrear duplicados na coluna "content"
content_counts = {}

# Contar os valores vazios, nulos e duplicados
for item in data:
    # Contando Titles
    if item['title'] == "":
        empty_titles += 1
    elif item['title'] is None:
        null_titles += 1
    
    # Contando Contents
    if item['content'] == "":
        empty_contents += 1
    elif item['content'] is None:
        null_contents += 1
    
    # Contar duplicados na coluna "content"
    content = item['content']
    if content in content_counts:
        content_counts[content] += 1
    else:
        content_counts[content] = 1

# Contar valores duplicados
duplicate_contents = sum(1 for count in content_counts.values() if count > 1)

total_titles = len(data)
total_contents = len(data) 

# Exibir os resultados
print(f"Total de Titles vazios: {empty_titles}")
print(f"Total de Titles nulos: {null_titles}")
print(f"Total de Contents vazios: {empty_contents}")
print(f"Total de Contents nulos: {null_contents}")
print(f"Total de Titles: {total_titles}")
print(f"Total de Contents: {total_contents}")
print(f"Total de Contents duplicados: {duplicate_contents}")

# Retirar as colunas "title" e "content" vazias
filtered_data = [item for item in data if item['title'] and item['content']]

# Remover linhas com valores duplicados na coluna "content"
seen_contents = set()
deduplicated_data = []
for item in filtered_data:
    if item['content'] not in seen_contents:
        deduplicated_data.append(item)
        seen_contents.add(item['content'])

# Contar duplicados após a remoção
deduplicated_content_counts = {}
for item in deduplicated_data:
    content = item['content']
    if content in deduplicated_content_counts:
        deduplicated_content_counts[content] += 1
    else:
        deduplicated_content_counts[content] = 1

remaining_duplicates = sum(1 for count in deduplicated_content_counts.values() if count > 1)

# Contagem final
empty_titles = 0
null_titles = 0
empty_contents = 0
null_contents = 0

for item in deduplicated_data:
    # Contando Titles
    if item['title'] == "":
        empty_titles += 1
    elif item['title'] is None:
        null_titles += 1
    
    # Contando Contents
    if item['content'] == "":
        empty_contents += 1
    elif item['content'] is None:
        null_contents += 1

total_titles = len(deduplicated_data)
total_contents = len(deduplicated_data) 

print(f"------------------------Após a limpeza------------------------")
# Exibir os resultados separados
print(f"Total de Titles vazios: {empty_titles}")
print(f"Total de Titles nulos: {null_titles}")
print(f"Total de Contents vazios: {empty_contents}")
print(f"Total de Contents nulos: {null_contents}")
print(f"Total de Titles: {total_titles}")
print(f"Total de Contents: {total_contents}")
print(f"Total de Contents duplicados restantes: {remaining_duplicates}")

# Salvar em um novo arquivo JSON
output_path = "./data/trn_sanitized_final.json"
with open(output_path, "w") as outfile:
    json.dump(deduplicated_data, outfile, indent=4)

print(f"Novo arquivo salvo em: {output_path}")
