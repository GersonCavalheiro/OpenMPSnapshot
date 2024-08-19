BEGIN {
    # Inicializa as variáveis no início do processamento
    parallel_for_found = 0;
    schedule_found = 0;
    static_found = 0;
    block_depth = 0;
    relevant_block_started = 0;
    if_found = 0;
}

FNR == 1 {
    # Reinicializa as variáveis para cada novo arquivo
    parallel_for_found = 0;
    schedule_found = 0;
    static_found = 0;
    block_depth = 0;
    relevant_block_started = 0;
    if_found = 0;
}

{
    # Verifica se a linha contém os padrões desejados
	    if (/^#pragma omp for/ && /schedule/ && /runtime/) {
        parallel_for_found = 1;
    }

    # Verifica se um bloco relevante começou
    if (parallel_for_found) {
        if (/\{/) {
            block_depth++;
            if (block_depth == 1 && !relevant_block_started) {
                relevant_block_started = 1;
                if_found = 0; # Reinicia a busca por 'if' dentro do novo bloco relevante
            }
        }
    }

    # Verifica o fechamento de blocos
    if (/\}/) {
        if (relevant_block_started) {
            block_depth--;
            if (block_depth == 0) {
                # Reseta as flags ao sair do bloco relevante
                parallel_for_found = 0;
                relevant_block_started = 0;
                if_found = 0;
            }
        }
    }

    # Verifica a presença de 'if' dentro do bloco relevante
    if (relevant_block_started && !if_found && /if/) {
        print FILENAME "," FNR;
        if_found = 1; # Para de procurar por 'if' após encontrar o primeiro
    }
}

