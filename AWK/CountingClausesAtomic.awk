    #!/usr/bin/awk -f

BEGIN {
    # Inicializa os contadores para cada padrão
    semNada = 0
    update = 0
    write = 0
    read = 0
    hint = 0
    capture = 0
    compare = 0
    outros = 0  # Contador para outros padrões
}

# Procura por linhas que começam com "#pragma omp atomic"
/^#pragma omp atomic/ { 
    # Verifica se a linha termina logo após o "atomic"
    if ($0 == "^#pragma omp atomic") { 
        semNada++
    } else {
        # Divide a linha em palavras para analisar o que vem depois de "atomic"
        split($0, palavras, " ")

        # Inicializa uma flag para indicar se algum padrão conhecido foi encontrado
        encontrouPadrao = 0 

        # Verifica cada palavra para identificar os padrões
        for (i in palavras) {
            if (palavras[i] == "update") { 
                update++
                encontrouPadrao = 1
            }
            if (palavras[i] == "write") { 
                write++
                encontrouPadrao = 1
            }
            if (palavras[i] == "read") { 
                read++
                encontrouPadrao = 1
            }
            if (palavras[i] == "hint") { 
                hint++
                encontrouPadrao = 1
            }
            if (palavras[i] == "capture") { 
                capture++
                encontrouPadrao = 1
            }
            if (palavras[i] == "compare") { 
                compare++
                encontrouPadrao = 1
            }
        }

        # Se nenhum padrão conhecido foi encontrado, incrementa o contador "outros"
        if (!encontrouPadrao) { 
            outros++
        }
    }
}

# Imprime os resultados no formato solicitado
END {
    printf("Update & write & read & compare & capture & hint\\\\ \n")
    printf("%d & %d & %d & %d & %d & %d\\\\\n", outros+update, write, read, compare, capture, hint)
    printf("Default: %d\n", outros)
    printf("Total: %d\n", outros+semNada+update + write + read + compare + capture + hint);
}

