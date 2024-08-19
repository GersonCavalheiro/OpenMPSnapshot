#include "hashtable.h"
#include "cell.h"
void life3d_run(unsigned int size, hashtable_t *state, unsigned int num_cells, unsigned long generations) {
hashtable_t *next_state;
#pragma omp parallel shared(state, num_cells, next_state)
for (unsigned long g = 0; g < generations; g++) {
#pragma omp master
{
next_state = HT_create(num_cells * 6);
num_cells = 0;
}
#pragma omp barrier
#pragma omp for reduction(+:num_cells)
for (unsigned int i = 0; i < state->capacity; i++) {
cell_t c = state->table[i];
if (c == 0) {
continue;
}
cell_t neighbors[6];
cell_get_neighbors(c, neighbors, size);
if (cell_next_state(c, neighbors, state)) {
HT_set_atomic(next_state, c);
num_cells++;
}
for (size_t j = 0; j < 6; j++) {
c = neighbors[j];
if (!(HT_contains(state, c)) && !(HT_contains(next_state, c))) {
cell_t buf[6];
cell_get_neighbors(c, buf, size);
if (cell_next_state(c, buf, state)) {
HT_set_atomic(next_state, c);
num_cells++;
}
}
}
}
#pragma omp master
{
HT_free(state);
state = next_state;
}
}
return;
}
