BEGIN {
    block_depth = 0;
    parallel_for_count = 0;
}

/#pragma omp parallel for/ {
    parallel_for_count++;

    if (parallel_for_count == 2) { # Segunda ocorrência no mesmo bloco
        print FILENAME "," FNR;
        # Não precisamos mais sair aqui
    }
}

/{/ {
    block_depth++;
}

/}/ {
    block_depth--;
    parallel_for_count = 0; # Reinicia a contagem ao final de cada bloco
}

