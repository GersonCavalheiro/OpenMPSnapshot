BEGIN {
    FS = "[ \t\n]+"  # Define o delimitador de campo como espaço, tabulação ou nova linha
    critical_found = 0
}

FNR == 1 {
    line_num = 1
}

# Detecta linhas que começam com "#pragma omp critical"
/^#pragma omp critical/ {
    critical_found = 1
    next
}

critical_found {
    # Verifica se a linha contém a variável no lado esquerdo da atribuição
    if ($0 ~ /^[ \t]*\{?[ \t]*([a-zA-Z_][a-zA-Z0-9_\[\]]*)[ \t]*=[ \t]*(.*);?[ \t]*\}?$/) {
        var = extract_variable($0)  # Chama a função para extrair a variável
        rhs = substr($0, index($0, "=") + 1)  # Obtém o lado direito da atribuição
        # Verifica se a variável está presente no lado direito da atribuição
        if (index(rhs, var) > 0) {
            print FILENAME ":" line_num ": " $0
        }
    }
    critical_found = 0
}

{
    line_num++
}

# Função para extrair a variável do lado esquerdo da atribuição
function extract_variable(line) {
    gsub(/^[ \t]*\{?[ \t]*/, "", line)  # Remove espaços e '{' no início
    match(line, /[a-zA-Z_][a-zA-Z0-9_\[\]]*/)  # Encontra a variável no lado esquerdo
    return substr(line, RSTART, RLENGTH)  # Retorna a variável encontrada
}

