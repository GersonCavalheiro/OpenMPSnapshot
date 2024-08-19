#!/usr/bin/awk -f

BEGIN {
    FS = "[ \t\n]+"
    critical_found = 0
    inside_block = 0
    total_incremento = 0
    total_atribuicao = 0
    total_leitura_escrita = 0
}

FNR == 1 {
    # Reinicia as variáveis para cada novo arquivo
    critical_found = 0
    inside_block = 0
    line_num = 0
}

/^#pragma omp critical/ {  # O ^ é mantido aqui para ancorar o #pragma omp critical
    critical_found = 1
    next
}

# Verifica se está dentro de um bloco ou se não há blocos E se critical_found está ativo
critical_found && (inside_block || !/\{|\}/) {
    # Verifica padrão de atribuição em array com todos os operadores compostos
    if (match($0, /([a-zA-Z_][a-zA-Z0-9_]*(\\[\w+])*)[ \t]*([+\-*/%&\|^<>]=)[ \t]*[^;]+;[ \t]*$/)) {
        array_assignment = $0
        assignment_line_num = FNR
        print FILENAME ":" assignment_line_num ":" array_assignment " Atribuição Composta"
        total_atribuicao++  
        next
    }

    # Verifica padrões de pré/pós-incremento/decremento (dentro ou fora de blocos), incluindo arrays
    if ($0 ~ /([a-zA-Z_][a-zA-Z0-9_]*(\\[\w+])*([\[][^\]]*[\]])?)(\+\+|\-\-);$/ || 
        $0 ~ /(\+\+|\-\-)([a-zA-Z_][a-zA-Z0-9_]*(\\[\w+])*([\[][^\]]*[\]])?);$/) {
        print FILENAME ":" line_num ":" $0 " Incremento"
        total_incremento++ 
        next
    }

    # Verifica padrão de leitura ou escrita simples
    if (match($0, /([a-zA-Z_][a-zA-Z0-9_]*(\\[\w+])*)[ \t]*=[ \t]*[^;]+;[ \t]*$/)) {
        simple_assignment = $0
        assignment_line_num = FNR
        print FILENAME ":" assignment_line_num ":" simple_assignment " Leitura ou Escrita Simples"
        total_leitura_escrita++ 
        next
    }
    
    # Desativa critical_found se nenhuma das condições acima for satisfeita
    critical_found = 0
}

# Lida com o início e o fim de blocos 
critical_found && /\{/ {
    inside_block = 1
    next
}

critical_found && inside_block && /\}/ {
    inside_block = 0
    next
}

{
    line_num++
}

END {
    print "Total de Incrementos:", total_incremento
    print "Total de Atribuições Compostas:", total_atribuicao
    print "Total de Leituras ou Escritas Simples:", total_leitura_escrita
}

