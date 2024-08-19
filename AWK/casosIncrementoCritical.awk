BEGIN {
    FS = "[ \t\n]+"
    critical_found = 0
}

FNR == 1 {
    line_num = 0
}

/^[^/]*#pragma omp critical/ {
    critical_found = 1
    next
}

critical_found && /^{/ {  # Início de bloco
    if (getline > 0) {  # Lê a próxima linha
        if ($0 ~ /^([a-zA-Z_][a-zA-Z0-9_]*(\\[\w+])*)(\+\+|\-\-);$/ || $0 ~ /^(\+\+|\-\-)([a-zA-Z_][a-zA-Z0-9_]*(\\[\w+])*);$/) {
            print FILENAME ":" line_num ":" $0  # Imprime o padrão encontrado
        }
    }
    critical_found = 0  # Reinicia após verificar a linha dentro do bloco
    next
}

critical_found && $1 ~ /^([a-zA-Z_][a-zA-Z0-9_]*(\\[\w+])*)(\+\+|\-\-);$/ {  # Pré-incremento/decremento sem bloco
    print FILENAME ":" line_num ":" $1
    critical_found = 0
    next
}

critical_found && $1 ~ /^(\+\+|\-\-)([a-zA-Z_][a-zA-Z0-9_]*(\\[\w+])*);$/ {  # Pós-incremento/decremento sem bloco
    print FILENAME ":" line_num ":" $1
    critical_found = 0
    next
}

{
    line_num++
}

