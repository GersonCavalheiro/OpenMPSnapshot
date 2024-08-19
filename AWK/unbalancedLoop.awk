#!/usr/bin/awk -f

BEGIN {
    FS = "[ \t\n]+"  # Define o separador como espaços em branco ou tabulação
    schedule_found = 0  # Variável de controle para indicar se encontramos uma diretiva OpenMP com schedule
    for_found = 0  # Variável de controle para indicar se encontramos um loop 'for' após a diretiva
    inside_block = 0  # Variável de controle para indicar se estamos dentro do bloco de comandos
    currentSchedule = ""  # Variável para armazenar o tipo de schedule encontrado

    # Inicializa os contadores para cada tipo de schedule
    none = 0
    static = 0
    dynamic = 0
    guided = 0
    auto = 0
    runtime = 0
}

# Padrões para identificar diretivas OpenMP com schedule
/^#pragma omp (parallel )?for.*schedule/ {
    # Extrai o tipo de schedule da linha da diretiva OpenMP
    if (match($0, /\((static|dynamic|guided|auto|runtime)/)) {
        currentSchedule = substr($0, RSTART + 1, RLENGTH - 1)
        schedule_found = 1
    } else {
        currentSchedule = ""
        schedule_found = 0
    }
    next  # Avança para a próxima linha
}

/^#pragma omp parallel for/ && !/schedule/ {
    currentSchedule = "none"
    schedule_found = 1
    next
}

# Verifica se a linha contém um loop 'for' após a diretiva OpenMP
schedule_found && /for\s*\(.+\)/ {
    for_found = 1
    next  # Avança para a próxima linha
}

# Entra no bloco de comandos se encontramos a diretiva e o loop 'for'
schedule_found && for_found && /\{/ {
    inside_block = 1  # Entrou no bloco de comandos
    schedule_found = 0  # Reinicia a variável de controle
    for_found = 0  # Reinicia a variável de controle
    next  # Avança para a próxima linha
}

# Conta ocorrências de comandos 'if' dentro do bloco de comandos
inside_block {
    if ($0 ~ /if\s*\(.+\)/) {
        if (currentSchedule == "static") static++
        else if (currentSchedule == "dynamic") dynamic++
        else if (currentSchedule == "guided") guided++
        else if (currentSchedule == "auto") auto++
        else if (currentSchedule == "runtime") runtime++
        else if (currentSchedule == "none") none++
    }
    if ($0 ~ /\}/) {
        inside_block = 0  # Sai do bloco de comandos
        currentSchedule = ""  # Reinicia o tipo de schedule
    }
}

# Imprime os resultados
END {
    print "None & static & dynamic & guided & auto & runtime \\\\"
    print none " & " static " & " dynamic " & " guided " & " auto " & " runtime "\\\\"
}

