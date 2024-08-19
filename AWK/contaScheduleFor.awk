#!/usr/bin/awk -f

BEGIN {
    # Inicializa os contadores para cada padrão de scheduling
    static = 0
    dynamic = 0
    guided = 0
    auto = 0
    runtime = 0
    static_chunk = 0
    dynamic_chunk = 0
    guided_chunk = 0
    auto_chunk = 0
    runtime_chunk = 0
    total = 0
    pstatic = 0
    pdynamic = 0
    pguided = 0
    pauto = 0
    pruntime = 0
    pstatic_chunk = 0
    pdynamic_chunk = 0
    pguided_chunk = 0
    pauto_chunk = 0
    pruntime_chunk = 0
    ptotal = 0
}

# Procura por linhas que iniciam com "#pragma omp parallel for" ou "#pragma omp for" e que também contenham a palavra "schedule"
{
    if ($0 ~ /^\s*#pragma omp (parallel )?for/ && $0 ~ /schedule/) {
        if ($0 ~ /schedule\(\s*static\s*\)/) {
           if ($0 ~ /parallel/) {
              pstatic++
	   } else static++
        }
        if ($0 ~ /schedule\s*\(\s*dynamic\s*\)/) {
           if ($0 ~ /parallel/) {
              pdynamic++
	   } else dynamic++
        }
        if ($0 ~ /schedule\s*\(\s*guided\s*\)/) {
           if ($0 ~ /parallel/) {
              pguided++
	   } else guided++
        }
        if ($0 ~ /schedule\s*\(\s*auto\s*\)/) {
           if ($0 ~ /parallel/) {
              pauto++
	   } else auto++
        }
        if ($0 ~ /schedule\s*\(\s*runtime\s*\)/) {
           if ($0 ~ /parallel/) {
              pruntime++
	   } else runtime++
        }
        if ($0 ~ /schedule\s*\(\s*static\s*,.+\)/) {
           if ($0 ~ /parallel/) {
              pstatic_chunk++
	   } else static_chunk++
        }
        if ($0 ~ /schedule\s*\(\s*dynamic\s*,.*\)/) {
           if ($0 ~ /parallel/) {
              pdynamic_chunk++
	   } else dynamic_chunk++
        }
        if ($0 ~ /schedule\s*\(\s*guided\s*,.*\)/) {
           if ($0 ~ /parallel/) {
              pguided_chunk++
	   } else guided_chunk++
        }
        if ($0 ~ /schedule\s*\(\s*auto\s*,.*\)/) {
           if ($0 ~ /parallel/) {
              pauto_chunk++
	   } else auto_chunk++
        }
        if ($0 ~ /schedule\s*\(\s*runtime\s*,\s*\w+\s*\)/) {
           if ($0 ~ /parallel/) {
              pruntime_chunk++
	   } else runtime_chunk++
        }
        if ($0 ~ /parallel/) {
           ptotal++
	} else ++total
    }
}

# Imprime os resultados
END {
    print "static:", static
    print "dynamic:", dynamic
    print "guided:", guided
    print "auto:", auto
    print "runtime:", runtime
    print "static_chunk:", static_chunk
    print "dynamic_chunk:", dynamic_chunk
    print "guided_chunk:", guided_chunk
    print "auto_chunk:", auto_chunk
    print "runtime_chunk:", runtime_chunk
    print "total:", total
    printf("parallel for & & %d/%d & %d/%d & %d/%d & %d/%d & %d/%d & %d\\\\\n", pstatic, pstatic_chunk, pdynamic, pdynamic_chunk, pguided, pguided_chunk, pauto, pauto_chunk, pruntime, pruntime_chunk, ptotal- pstatic- pstatic_chunk- pdynamic- pdynamic_chunk- pguided- pguided_chunk- pauto- pauto_chunk- pruntime- pruntime_chunk)
    printf("for & & %d/%d & %d/%d & %d/%d & %d/%d & %d/%d & %d\\\\\n", static, static_chunk, dynamic, dynamic_chunk, guided, guided_chunk, auto, auto_chunk, runtime, runtime_chunk, total- static- static_chunk- dynamic- dynamic_chunk- guided- guided_chunk- auto- auto_chunk- runtime- runtime_chunk)
}

