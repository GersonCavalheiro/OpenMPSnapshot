BEGIN {
    FS = "[ \t\n;=]+"
    critical_found = 0
    inside_critical_block = 0
    assignment_detected = 0
}

/^#pragma omp critical/ {
    critical_found = 1
    next
}

critical_found && /\{/ {
    inside_critical_block = 1
    next
}

critical_found && inside_critical_block && /\}/ {
    inside_critical_block = 0
    critical_found = 0
    assignment_detected = 0
    next
}

critical_found && !inside_critical_block && match($0, /^[ \t]*([a-zA-Z_][a-zA-Z0-9_]*\[[^]]+\])[ \t]*=[ \t]*.*\1.*;[ \t]*$/) {
    array_assignment = $0
    assignment_detected = 1
    next
}

assignment_detected {
    print FILENAME ":" FNR ":" array_assignment
    assignment_detected = 0
}

