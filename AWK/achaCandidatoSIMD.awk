BEGIN {
    pragma_found = 0;
    for_found = 0;
}

/#pragma omp (parallel for|for)/ {
    pragma_found = 1;
    next;
}

/{/ {
    if (pragma_found) {
        next; # Pula para a próxima linha se encontrar a chave de abertura
    }
}

pragma_found && /for/ {
    for_found = 1;
    next;
}

for_found && /[a-zA-Z_]\w*\[([0-9]+|[a-zA-Z_]\w*)\]\s*=/ {
    print FILENAME "," FNR ":" $0;
    pragma_found = 0;
    for_found = 0;
}

# Reinicia as flags se a próxima linha não corresponder ao padrão
{
    pragma_found = 0;
    for_found = 0;
}

