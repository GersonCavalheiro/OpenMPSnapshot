BEGIN {
    FS = "[ \t\n;=]+"
    critical_found = 0
}

/^#pragma omp critical/ {
    critical_found = 1
    next
}

{
    if (critical_found && match($0, /^[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)[ \t]*=[ \t]*.*;[ \t]*/)) {
        array_assignment = $0
        assignment_detected = 1
    } 
    critical_found = 0
    if (assignment_detected) {
        print FILENAME ":" FNR ":" array_assignment
        assignment_detected = 0
    }
}
