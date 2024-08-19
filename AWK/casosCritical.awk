#!/usr/bin/awk -f

BEGIN {
    FS = "[ \t\n]+"
    critical_found = 0
    inside_block = 0
    total_incremento = 0
    total_atribuicao = 0
    total_leitura_escrita = 0
    total_while = 0
    total_for = 0
    total_if = 0
}

FNR == 1 {
    # Reinicia as variáveis para cada novo arquivo
    critical_found = 0
    inside_block = 0
    line_num = 0
}

/^#pragma omp critical/ {
    critical_found = 1
    next
}

# Verifica se está dentro de um bloco ou se não há blocos E se critical_found está ativo
critical_found && (inside_block || !/\{|\}/) {
    # Verifica padrão de atribuição em array com todos os operadores compostos
    if (match($0, /([a-zA-Z_][a-zA-Z0-9_]*(\\[\w+])*)[ \t]*([+\-*/%&\|^<>]=).*;/)) {
        array_assignment = $0
        assignment_line_num = FNR
        print FILENAME ":" assignment_line_num ":" array_assignment " Atribuição Composta"
        total_atribuicao++
        next
    }

    # Verifica padrões de pré/pós-incremento/decremento (dentro ou fora de blocos), incluindo arrays
    if ($0 ~ /([a-zA-Z_][a-zA-Z0-9_]*(\[\.+\])*)[ \t]*(\+\+|\-\-);/ ||
        $0 ~ /(\+\+|\-\-)[ \t]*([a-zA-Z_][a-zA-Z0-9_]*(\[\.+\])*);/) {
        print FILENAME ":" line_num ":" $0 " Incremento"
        total_incremento++
        next
    }

    # Verifica padrão de leitura ou escrita simples
    if (match($0, /[a-zA-Z_][a-zA-Z0-9_]*(\[\.+\])*[ \t]*=/)) {
        simple_assignment = $0
        assignment_line_num = FNR
        print FILENAME ":" assignment_line_num ":" simple_assignment " Leitura ou Escrita Simples"
        total_leitura_escrita++
        next
    }

    # Desativa critical_found se nenhuma das condições acima for satisfeita
    critical_found = 0
}

# Verifica se a linha seguinte contém abertura de bloco
critical_found && getline > 0 && $0 ~ /\{/ {
    inside_block = 1

    # Reinicia os contadores para o bloco atual
    block_start_line = FNR
    block_total_while = 0
    block_total_for = 0
    block_total_if = 0

    while (getline > 0 && $0 !~ /\}/) {
        if ($0 ~ /while[ \t]*\(/) {
            block_total_while++
        }
        if ($0 ~ /for[ \t]*\(/) {
            block_total_for++
        }
        if ($0 ~ /if[ \t]*\(/) {
            block_total_if++
        }
    }

    # Acumula os totais do bloco nos totais gerais
    total_while += block_total_while
    total_for += block_total_for
    total_if += block_total_if

    inside_block = 0
}

{
    line_num++
}

END {
    print "Total de Incrementos:", total_incremento
    print "Total de Atribuições Compostas:", total_atribuicao
    print "Total de Leituras ou Escritas Simples:", total_leitura_escrita
    print "Total de While em Bloco:", total_while
    print "Total de For em Bloco:", total_for
    print "Total de If em Bloco:", total_if
}

