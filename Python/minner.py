import base64
from github import Github
import pandas as pd
import time
from datetime import datetime
import os
import logging

# Captura a entrada do usuário
user_input = int(input("Digite 0 para C ou 1 para C++: "))

# Verifica se a entrada do usuário é 0 ou 1
if user_input == 0:
    query = "topic:openmp language:C"
elif user_input == 1:
    query = "topic:openmp language:C++"
else:
    query = "topic:openmp language:C"
    print("Escolha inválida. Buscando por repositórios em C.")

# Configuração do logger
logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

# Token de autenticacao
g = Github("INSIRA AQUI SEU TOKEN DE ACESSO PESSOAL")

# Lista para armazenar os dados dos repositórios
repos_list = []

logging.info("Query utilizada: [" + query + "]")
print("Query utilizada: [" + query + "]")

caminho_atual = os.getcwd()
subpasta = "tcc_dados"
pasta_temporaria = "temp"
dir_name = os.path.join(caminho_atual, subpasta)
dir_temp = os.path.join(caminho_atual, pasta_temporaria)

# Inicia o temporizador
tempo_inicial = time.time()

repos = g.search_repositories(query=query, sort='stars', order='desc')
i = 0
j = 0
# Realiza a busca
# Percorre a lista de repositórios retornados pela busca
imprimir = True
for repo in repos:
    # Obtém o limite de taxa atual
    rate_limit = g.get_rate_limit()
    # Verifica se o limite de taxa está abaixo de um limite seguro
    if rate_limit.core.remaining < 100:
        reset_time = datetime.fromtimestamp(g.rate_limiting_resettime)
        # Obter o tempo atual em segundos
        current_time = datetime.now().timestamp()
        # Calcular o tempo restante em segundos
        wait_time = reset_time.timestamp() - current_time + 15
        # Verificar se há necessidade de aguardar
        if wait_time > 0:
            print(f"Limite de taxa excedido. Aguardando {wait_time} segundos...")
            logging.info(f"Limite de taxa excedido. Aguardando {wait_time} segundos...")
            time.sleep(wait_time)
    print("Taxa de limite atual: " + str(rate_limit) + "\n")
    logging.info("Taxa de limite atual: " + str(rate_limit) + "\n")
    faz_download = False
    repo_down = True
    # Extrai as informações relevantes de cada repositório
    if imprimir:
        print("Quantidade total de repos: " + str('{:,.2f}'.format(repos.totalCount)))
        logging.info("Quantidade total de repos: " + str('{:,.2f}'.format(repos.totalCount)))
        imprimir = False
    id = repo.id
    name = repo.name
    linguagens = repo.get_languages().keys()
    description = repo.description
    stars = repo.stargazers_count
    forks = repo.forks_count
    url = repo.html_url
    files_with_pragma = []
    
    # Diretório para salvar o repositório temporariamente
    repo_tmp_dir = os.path.join(dir_temp, str(id))
    os.makedirs(repo_tmp_dir, exist_ok=True)

    # Clona o repositório temporariamente
    git_url = repo.clone_url
    os.system(f"git clone {git_url} {repo_tmp_dir}")
    for root, dirs, files in os.walk(repo_tmp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        content_bytes = f.read()
                    # Adiciona o padding manualmente, se necessário
                    content_bytes = base64.b64encode(content_bytes)
                    content = base64.b64decode(content_bytes)
                    # Decodifica a string Base64
                    content = content.decode("utf-8", errors="ignore")
                    # Verifica se o arquivo possui a diretiva "#pragma"
                    if "pragma" in content:
                        if repo_down:
                            # Diretório para salvar os arquivos do repositório atual
                            repo_dir = os.path.join(dir_name, str(id))
                            os.makedirs(repo_dir, exist_ok=True)
                            files_with_pragma.append(file_path)
                            # Obtém o caminho completo do arquivo
                            file_path = os.path.join(repo_dir, file)
                            # Salva o arquivo no diretório
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(content)
                except (UnicodeDecodeError, IOError):
                        continue
    # Remove o repositório temporário
    os.system(f"rm -rf {repo_tmp_dir}")
    # Adiciona as informações em uma linha da lista
    repos_list.append([id,name, description, stars, forks, url, linguagens,files_with_pragma])
    j += 1
    print("[Repositorio: " + str(j) + "]\n[id: " + str(id) + "]\n[descricao: " + str(description) + "]")
    logging.info("[Repositorio: " + str(j) + "]\n[id: " + str(id) + "]\n[descricao: " + str(description) + "]")
tempo_final = time.time()
# Calcula e imprime o tempo de execucao
tempo_total = tempo_final - tempo_inicial
logging.info(f"Tempo de execução: {tempo_total} segundos")
print(f"Tempo de execução: {tempo_total} segundos")
df = pd.DataFrame(repos_list, columns=['id','Nome', 'Descrição', 'Estrelas', 'Forks', 'URL', 'Linguagens', 'Arquivos com PRAGMA'])
df.to_csv('repos_openmp.csv', index=False)
