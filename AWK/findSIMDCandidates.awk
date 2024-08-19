BEGIN {
    FS = "[ \t\n]+"
    pragma_found = 0
    line_num = 0
}

FNR == 1 {
    # Reinicia as variáveis para cada novo arquivo
    pragma_found = 0
    line_num = 0
}

/^[^/]*#pragma omp (parallel for|for)/ && !/simd/ {
    pragma_found = 1
    pragma_line = FNR
    next
}

pragma_found && /^for[ \t]*\(/ {
    if (getline > 0) {
        if ($0 ~ /^[^/]*[a-zA-Z_][a-zA-Z0-9_]*\[[^]]+\][ \t]*=[ \t]*.+/) {  # Verifica a atribuição de vetor
            print FILENAME ":" FNR
        }
    }
    pragma_found = 0
    next
}

{
    line_num++
}

