BEGIN {
    FS = "[ \t\n]+"
    critical_found = 0
    inside_block = 0
}

FNR == 1 {
    # Reinicia as variáveis para cada novo arquivo
    critical_found = 0
    inside_block = 0
}

/^#pragma omp critical/ {
    critical_found = 1
    next
}

critical_found && /\{/ {
    inside_block = 1
    next
}

critical_found && inside_block && /\}/ {
    inside_block = 0
    next
}

critical_found && (inside_block || !/\{|\}/) && match($0, /^[ \t]*([a-zA-Z_][a-zA-Z0-9_]*\[[^]]+\])[ \t]*=[ \t]*.*\1.*;[ \t]*$/) {
    # Verifica se está dentro de um bloco ou se não há blocos e se a linha contém uma atribuição válida
    array_assignment = $0
    assignment_line_num = FNR
    print FILENAME ":" assignment_line_num ":" array_assignment
    next
}

{
    # Atualiza o número da linha
    line_num = FNR
}

