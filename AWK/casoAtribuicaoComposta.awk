BEGIN {
    FS = "[ \t\n]+"
    critical_found = 0
}

/^#pragma omp critical/ {
    critical_found = 1
    next
}

FNR == 1 {
    line_num = 0
}

{
    line_num++
    if (critical_found) {
        # Verifica se a linha contém um operador de atribuição composto
        if (match($0, /^[ \t]*[{]?[ \t]*[a-zA-Z_][a-zA-Z0-9_\[\]]*[ \t]*[\+\-\*\/\%\&\|\^]=[ \t]*.*;[ \t]*[}]?[ \t]*$/)) {
            print FILENAME ":" line_num ":" $0
        }
        critical_found = 0
    }
}

