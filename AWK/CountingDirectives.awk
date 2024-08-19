BEGIN {
    directives["parallel"] = 0;
    directives["parallel for"] = 0;
    directives["for"] = 0;
    directives["parallel sections"] = 0;
    directives["sections"] = 0;
    directives["section"] = 0;
    directives["barrier"] = 0;
    directives["critical"] = 0;
    directives["master"] = 0;
    directives["single"] = 0;
    directives["flush"] = 0;
    directives["ordered"] = 0;
    directives["threadprivate"] = 0;
    directives["atomic"] = 0;
    directives["taskgroup"] = 0;
    directives["taskyield"] = 0;
    directives["task"] = 0;
    directives["taskwait"] = 0;
    directives["taskloop"] = 0;
    directives["target"] = 0;
    directives["simd"] = 0;
    directives["taskloop simd"] = 0;
    directives["parallel for simd"] = 0;
    directives["for simd"] = 0;
    directives["teams"] = 0;
    directives["distribute"] = 0;
    directives["requires"] = 0;
    directives["loop"] = 0;
    directives["allocate"] = 0;
    directives["depobj"] = 0;
    directives["metadirective"] = 0;
    directives["declare"] = 0;
    directives["mask"] = 0;
    directives["scope"] = 0;
    directives["assume"] = 0;
    directives["tile"] = 0;
    directives["error"] = 0;

    versions["parallel"] = "1.0";
    versions["parallel for"] = "1.0";
    versions["for"] = "1.0";
    versions["parallel sections"] = "1.0";
    versions["sections"] = "1.0";
    versions["section"] = "1.0";
    versions["barrier"] = "1.0";
    versions["critical"] = "1.0";
    versions["master"] = "1.0";
    versions["single"] = "1.0";
    versions["flush"] = "2.0";
    versions["ordered"] = "2.0";
    versions["threadprivate"] = "2.0";
    versions["atomic"] = "2.0";
    versions["taskgroup"] = "3.3";
    versions["taskyield"] = "3.4";
    versions["task"] = "3.0";
    versions["taskwait"] = "3.1";
    versions["taskloop"] = "3.2";
    versions["target"] = "4.0";
    versions["simd"] = "4.0";
    versions["taskloop simd"] = "4.5";
    versions["parallel for simd"] = "4.5";
    versions["for simd"] = "4.5";
    versions["teams"] = "4.5";
    versions["distribute"] = "4.5";
    versions["requires"] = "4.5";
    versions["loop"] = "5.0";
    versions["allocate"] = "5.0";
    versions["depobj"] = "5.0";
    versions["metadirective"] = "5.0";
    versions["declare"] = "5.0";
    versions["mask"] = "5.1";
    versions["scope"] = "5.1";
    versions["assume"] = "5.1";
    versions["tile"] = "5.1";
    versions["error"] = "5.1";

    categories["parallel"] = "Parallel Control";
    categories["parallel for"] = "Parallel Control; Loop";
    categories["for"] = "Parallel Control; Loop";
    categories["parallel sections"] = "Parallel Control";
    categories["sections"] = "Parallel Control";
    categories["section"] = "Parallel Control";
    categories["barrier"] = "Synchronization";
    categories["critical"] = "Synchronization";
    categories["master"] = "Parallel Control";
    categories["single"] = "Parallel Control";
    categories["flush"] = "Synchronization";
    categories["ordered"] = "Synchronization";
    categories["threadprivate"] = "Data Privacy and Sharing";
    categories["atomic"] = "Synchronization";
    categories["taskgroup"] = "Synchronization; Task";
    categories["taskyield"] = "Synchronization; Task";
    categories["task"] = "Parallel Control; Task";
    categories["taskwait"] = "Synchronization; Task";
    categories["taskloop"] = "Parallel Control; Loop; Task";
    categories["target"] = "Teams and Distribution";
    categories["simd"] = "Parallel Control; Loop; SIMD";
    categories["taskloop simd"] = "Parallel Control; Loop; Task; SIMD";
    categories["parallel for simd"] = "Parallel Control; Loop; SIMD";
    categories["for simd"] = "Parallel Control; Loop; SIMD";
    categories["teams"] = "Teams and Distribution";
    categories["distribute"] = "Parallel Control";
    categories["requires"] = "Metaprogramming and Requirements";
    categories["loop"] = "Parallel Control; Loop";
    categories["allocate"] = "Memory Allocation and Management";
    categories["depobj"] = "Memory Allocation and Management";
    categories["metadirective"] = "Metaprogramming and Requirements";
    categories["declare"] = "Metaprogramming and Requirements";
    categories["mask"] = "Execution Control and Debugging";
    categories["scope"] = "Execution Control and Debugging";
    categories["assume"] = "Metaprogramming and Requirements";
    categories["tile"] = "Parallel Control; Loop";
    categories["error"] = "Execution Control and Debugging";
}

/^#pragma omp parallel/ { directives["parallel"]++; }
/^#pragma omp parallel for/ { directives["parallel for"]++; }
/^#pragma omp for/ { directives["for"]++; }
/^#pragma omp parallel sections/ { directives["parallel sections"]++; }
/^#pragma omp sections/ { directives["sections"]++; }
/^#pragma omp section/ { directives["section"]++; }
/^#pragma omp barrier/ { directives["barrier"]++; }
/^#pragma omp critical/ { directives["critical"]++; }
/^#pragma omp master/ { directives["master"]++; }
/^#pragma omp single/ { directives["single"]++; }
/^#pragma omp flush/ { directives["flush"]++; }
/^#pragma omp ordered/ { directives["ordered"]++; }
/^#pragma omp threadprivate/ { directives["threadprivate"]++; }
/^#pragma omp atomic/ { directives["atomic"]++; }
/^#pragma omp taskgroup/ { directives["taskgroup"]++; }
/^#pragma omp taskyield/ { directives["taskyield"]++; }
/^#pragma omp task/ { directives["task"]++; }
/^#pragma omp taskwait/ { directives["taskwait"]++; }
/^#pragma omp taskloop/ { directives["taskloop"]++; }
/^#pragma omp target/ { directives["target"]++; }
/^#pragma omp simd/ { directives["simd"]++; }
/^#pragma omp taskloop simd/ { directives["taskloop simd"]++; }
/^#pragma omp parallel for simd/ { directives["parallel for simd"]++; }
/^#pragma omp for simd/ { directives["for simd"]++; }
/^#pragma omp teams/ { directives["teams"]++; }
/^#pragma omp distribute/ { directives["distribute"]++; }
/^#pragma omp requires/ { directives["requires"]++; }
/^#pragma omp loop/ { directives["loop"]++; }
/^#pragma omp allocate/ { directives["allocate"]++; }
/^#pragma omp depobj/ { directives["depobj"]++; }
/^#pragma omp metadirective/ { directives["metadirective"]++; }
/^#pragma omp declare/ { directives["declare"]++; }
/^#pragma omp mask/ { directives["mask"]++; }
/^#pragma omp scope/ { directives["scope"]++; }
/^#pragma omp assume/ { directives["assume"]++; }
/^#pragma omp tile/ { directives["tile"]++; }
/^#pragma omp error/ { directives["error"]++; }

END { # Executado ao final de cada arquivo
    for (directive in directives) {
        #if (directives[directive] > 0) { # Imprime apenas se a diretiva foi encontrada
            printf("%s & %d & %s  & %s \\\\\n", 
                   directive, directives[directive], versions[directive], categories[directive]);
        #}
    }
    # Limpar os contadores para o próximo arquivo
    for (directive in directives) {
        directives[directive] = 0;
    }
}

