#pragma omp target teams num_teams((NUM*NUM/2)/BLOCK_SIZE) thread_limit(BLOCK_SIZE)
{
Real max_cache[BLOCK_SIZE];
#pragma omp parallel
{
int lid = omp_get_thread_num();
int tid = omp_get_team_num();
int gid = tid * BLOCK_SIZE + lid;
int row = (gid % (NUM/2)) + 1; 
int col = (gid / (NUM/2)) + 1; 

max_cache[lid] = ZERO;

int NUM_2 = NUM >> 1;
Real new_v = ZERO;

if (row != NUM_2) {
Real p_ij, p_ijp1, new_v2;

p_ij = pres_red(col, row);
p_ijp1 = pres_black(col, row + ((col + 1) & 1));

new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
v(col, (2 * row) - (col & 1)) = new_v;


p_ij = pres_black(col, row);
p_ijp1 = pres_red(col, row + (col & 1));

new_v2 = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
v(col, (2 * row) - ((col + 1) & 1)) = new_v2;


new_v = fmax(fabs(new_v), fabs(new_v2));

if (col == NUM) {
new_v = fmax(new_v, fabs( v(NUM + 1, (2 * row)) ));
}

} else {

if ((col & 1) == 1) {
Real p_ij = pres_red(col, row);
Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));

new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
v(col, (2 * row) - (col & 1)) = new_v;

} else {
Real p_ij = pres_black(col, row);
Real p_ijp1 = pres_red(col, row + (col & 1));

new_v = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
v(col, (2 * row) - ((col + 1) & 1)) = new_v;
}

new_v = fabs(new_v);

new_v = fmax(fabs( v(col, NUM) ), new_v);
new_v = fmax(fabs( v(col, 0) ), new_v);

new_v = fmax(fabs( v(col, NUM + 1) ), new_v);

} 

max_cache[lid] = new_v;

#pragma omp barrier

int i = BLOCK_SIZE >> 1;
while (i != 0) {
if (lid < i) {
max_cache[lid] = fmax(max_cache[lid], max_cache[lid + i]);
}
#pragma omp barrier
i >>= 1;
}

if (lid == 0) {
max_v_arr[tid] = max_cache[0];
}
}
}
