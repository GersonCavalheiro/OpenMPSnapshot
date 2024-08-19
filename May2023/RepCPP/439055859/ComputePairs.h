#ifndef COMPUTE_PAIRS_H
#define COMPUTE_PAIRS_H


void compute_pairs(Grid* grid,
Float rmin_short,
Float rmax_short,
Float rmin_long,
Float rmax_long,
Float rmin_cf,
Float rmax_cf,
int np) {
int maxsep_short = ceil(rmax_short / grid->cellsize);  
int maxsep_long_or_cf = ceil(fmax(rmax_long, rmax_cf) / grid->cellsize);  
int ne;
Float rmax_short2 = rmax_short * rmax_short;
Float rmin_short2 = rmin_short * rmin_short;  
Float rmax_long2 = rmax_long * rmax_long;
Float rmin_long2 = rmin_long * rmin_long;
Float rmax_cf2 = rmax_cf * rmax_cf;
Float rmin_cf2 = rmin_cf * rmin_cf;
uint64 cnt = 0, cnt2 = 0, cnt3 = 0;

Pairs* pairs_i = new Pairs[np];


STimer accpairs,
powertime;  

accpairs.Start();

#ifdef OPENMP
#pragma omp parallel for schedule(dynamic, 8) reduction(+ : cnt)
#endif

for (ne = 0; ne < grid->nf; ne++) {
int n = grid->filled[ne];  

#ifdef OPENMP
int thread = omp_get_thread_num();
assert(omp_get_num_threads() <= MAXTHREAD);
if (ne == 0)
printf("# Running on %d threads.\n", omp_get_num_threads());
#else
int thread = 0;
if (ne == 0)
printf("# Running single threaded.\n");
#endif
if (int(ne % 1000) == 0)
printf("Computing cell %d of %d on thread %d (1st loop)\n", ne, grid->nf, thread);

Cell primary = grid->c[n];
integer3 prim_id = grid->cell_id_from_1d(n);

for (int j = primary.start; j < primary.start + primary.np; j++) {
Float primary_w = grid->p[j].w;

#if (PERIODIC)
for (int bin = 0; bin < NBIN_SHORT; bin++) {
Float rmin_bin = rmin_short + bin * (rmax_short - rmin_short) / NBIN_SHORT;
Float rmax_bin = rmin_short + (bin + 1) * (rmax_short - rmin_short) / NBIN_SHORT;
pairs_i[j].add(bin, primary_w * (grid->sumw_neg) / pow(grid->max_boxsize, 3) * 4 * M_PI / 3 * (pow(rmax_bin, 3) - pow(rmin_bin, 3)));
}
#endif

integer3 delta;
for (delta.x = -maxsep_short; delta.x <= maxsep_short; delta.x++)
for (delta.y = -maxsep_short; delta.y <= maxsep_short; delta.y++)
for (delta.z = -maxsep_short; delta.z <= maxsep_short; delta.z++) {
const int samecell = (delta.x == 0 && delta.y == 0 && delta.z == 0) ? 1 : 0;

int tmp_test = grid->test_cell(prim_id + delta);
if (tmp_test < 0)
continue;
Cell sec = grid->c[tmp_test];

Float3 ppos = grid->p[j].pos;
#if PERIODIC
ppos -= grid->cell_sep(delta);
#endif

for (int k = sec.start; k < sec.start + sec.np;
k++) {
if (samecell && j == k)
continue;  
Float3 dx = grid->p[k].pos - ppos;
Float norm2 = dx.norm2();
if (norm2 < rmax_short2 && norm2 > rmin_short2)
cnt++;
else
continue;

norm2 = sqrt(norm2);  
int bin = floor((norm2 - rmin_short) / (rmax_short - rmin_short) * NBIN_SHORT);

#if (PERIODIC)
if (grid->p[k].w > 0) 
#endif
pairs_i[j].add(bin, grid->p[k].w * primary_w);
npcf[thread].excl_3pcf(bin, grid->p[k].w * grid->p[k].w * primary_w);

#if (!PREVENT_TRIANGLES && !IGNORE_TRIANGLES)
if (rmin_long < 2*rmax_short) {
integer3 delta2;
for (delta2.x = -maxsep_short; delta2.x <= maxsep_short; delta2.x++)
for (delta2.y = -maxsep_short; delta2.y <= maxsep_short; delta2.y++)
for (delta.z = -maxsep_short; delta.z <= maxsep_short; delta2.z++) {
int tmp_test = grid->test_cell(prim_id + delta2);
if (tmp_test < 0)
continue;
Cell third = grid->c[tmp_test];

Float3 ppos2 = grid->p[j].pos;
#if PERIODIC
ppos2 -= grid->cell_sep(delta2);
#endif

Float3 spos2 = grid->p[k].pos;
#if PERIODIC
spos2 -= grid->cell_sep(delta2) - grid->cell_sep(delta);
#endif
for (int l = third.start; l < third.start + third.np;
l++) {
if ((j==l) || (k==l))
continue;  
Float3 dx = grid->p[l].pos - ppos2;
Float norm_2 = dx.norm2();
Float3 dx_l = grid->p[l].pos - spos2;
Float norm_l2 = dx_l.norm2();
if (norm_2 < rmax_short2 && norm_2 > rmin_short2 && norm_l2 < rmax_long2 && norm_l2 > rmin_long2)
cnt3++;
else
continue;

norm_2 = sqrt(norm_2);  
norm_l2 = sqrt(norm_l2);
int bin2 = floor((norm_2 - rmin_short) / (rmax_short - rmin_short) * NBIN_SHORT);
int bin_long = floor((norm_l2 - rmin_long) / (rmax_long - rmin_long) * NBIN_LONG);

npcf[thread].excl_4pcf_triangle(bin_long, bin, bin2,
grid->p[l].w * grid->p[k].w * primary_w * primary_w);
}  
}      
}
#endif

#if (!PREVENT_TRIANGLES && !IGNORE_TRIANGLES)
if (rmin_long < rmax_short) {
if (norm2 >= rmax_long2 || norm2 <= rmin_long2)
continue;
int bin_long = floor((norm2 - rmin_long) / (rmax_long - rmin_long) * NBIN_LONG);
npcf[thread].excl_4pcf_tripleside(bin_long, bin, grid->p[k].w * primary_w);
}
#endif
}  
}      
pairs[thread].sum_power(pairs_i + j);

#if (!PREVENT_TRIANGLES && !IGNORE_TRIANGLES)
if (rmin_long < rmax_short) {
for (delta.x = -maxsep_short; delta.x <= maxsep_short; delta.x++)
for (delta.y = -maxsep_short; delta.y <= maxsep_short; delta.y++)
for (delta.z = -maxsep_short; delta.z <= maxsep_short; delta.z++) {
const int samecell =
(delta.x == 0 && delta.y == 0 && delta.z == 0)
? 1
: 0;

int tmp_test = grid->test_cell(prim_id + delta);
if (tmp_test < 0)
continue;
Cell sec = grid->c[tmp_test];

Float3 ppos = grid->p[j].pos;
#if PERIODIC
ppos -= grid->cell_sep(delta);
#endif

for (int k = sec.start; k < sec.start + sec.np;
k++) {
if (samecell && j == k)
continue;  
Float3 dx = grid->p[k].pos - ppos;
Float norm2 = dx.norm2();
if (norm2 >= rmax_short2 || norm2 <= rmin_short2)
continue;

norm2 = sqrt(norm2);  
int bin = floor((norm2 - rmin_short) / (rmax_short - rmin_short) * NBIN_SHORT);

dx = dx / norm2;


if (norm2 >= rmax_long2 || norm2 <= rmin_long2)
continue;
int bin_long = floor((norm2 - rmin_long) / (rmax_long - rmin_long) * NBIN_LONG);
npcf[thread].excl_4pcf_doubleside(pairs_i + j, bin_long, bin, grid->p[k].w * primary_w);
}  
}      
}
#endif

npcf[thread].add_3pcf(pairs_i + j, primary_w);
}  

}  

accpairs.Stop();


powertime.Start();

#ifdef OPENMP
#pragma omp parallel for schedule(dynamic, 8) reduction(+ : cnt)
#endif

for (ne = 0; ne < grid->nf; ne++) {
int n = grid->filled[ne];  

#ifdef OPENMP
int thread = omp_get_thread_num();
assert(omp_get_num_threads() <= MAXTHREAD);
#else
int thread = 0;
#endif
if (int(ne % 1000) == 0)
printf("Computing cell %d of %d on thread %d (2nd loop)\n", ne, grid->nf,
thread);

Cell primary = grid->c[n];
integer3 prim_id = grid->cell_id_from_1d(n);

for (int j = primary.start; j < primary.start + primary.np; j++) {
Float primary_w = grid->p[j].w;
integer3 delta;


for (delta.x = -maxsep_long_or_cf; delta.x <= maxsep_long_or_cf;
delta.x++)
for (delta.y = -maxsep_long_or_cf; delta.y <= maxsep_long_or_cf;
delta.y++)
for (delta.z = -maxsep_long_or_cf; delta.z <= maxsep_long_or_cf;
delta.z++) {
int tmp_test = grid->test_cell(prim_id + delta);
if (tmp_test < 0)
continue;
Cell sec = grid->c[tmp_test];

Float3 ppos = grid->p[j].pos;
#if PERIODIC
ppos -= grid->cell_sep(delta);
#endif

int end = sec.start + sec.np; 
if (j < end) end = j; 

for (int k = sec.start; k < end; k++) {
Float3 dx = grid->p[k].pos - ppos;
Float norm2 = dx.norm2();
if ((norm2 < rmax_long2 && norm2 > rmin_long2) || (norm2 < rmax_cf2 && norm2 > rmin_cf2))
cnt2++;
else
continue;

norm2 = sqrt(norm2);  
dx = dx / norm2; 
int bin_long = floor((norm2 - rmin_long) / (rmax_long - rmin_long) * NBIN_LONG);
if ((bin_long >= 0) && (bin_long < NBIN_LONG)) { 
npcf[thread].add_4pcf(bin_long, pairs_i + j, pairs_i + k);
npcf[thread].add_3pcf_mixed(bin_long, pairs_i + j, grid->p[k].w);
npcf[thread].add_3pcf_mixed(bin_long, pairs_i + k, primary_w); 
#if (!PREVENT_TRIANGLES && !IGNORE_TRIANGLES)
int bin_short = floor((norm2 - rmin_short) / (rmax_short - rmin_short) * NBIN_SHORT);
if ((bin_short >= 0) && (bin_short < NBIN_SHORT))
npcf[thread].excl_3pcf_mixed(bin_long, bin_short, primary_w * grid->p[k].w * (primary_w + grid->p[k].w));
#endif
#if (PERIODIC)
if ((primary_w > 0) && (grid->p[k].w > 0)) 
#endif
npcf[thread].add_2pcf_long(bin_long, primary_w * grid->p[k].w);
}
int bin_cf = floor((norm2 - rmin_cf) / (rmax_cf - rmin_cf) * NBIN_CF);
#if (PERIODIC)
Float mu = dx.z; 
#else
Float3 sumx = grid->p[k].pos + ppos;
Float sumx_norm = sumx.norm();
sumx = sumx / sumx_norm;
Float mu = sumx.dot(dx); 
#endif
#if (PERIODIC)
if ((primary_w > 0) && (grid->p[k].w > 0)) 
#endif
if ((bin_cf >= 0) && (bin_cf < NBIN_CF)) 
finepairs[thread].add(bin_cf, mu, grid->p[k].w * primary_w);
}  
}      
}  

}  

#if (PERIODIC)
for (int bin = 0; bin < NBIN_CF; bin++) {
Float rmin_bin = rmin_cf + bin * (rmax_cf - rmin_cf) / NBIN_CF;
Float rmax_bin = rmin_cf + (bin + 1) * (rmax_cf - rmin_cf) / NBIN_CF;
Float w_prod = 0.5 * grid->sumw_neg * (2 * grid->sumw_pos + grid->sumw_neg) / pow(grid->max_boxsize, 3) * 4 * M_PI / 3 * (pow(rmax_bin, 3) - pow(rmin_bin, 3)) / MBIN_CF; 
for (int mubin = 0; mubin < MBIN_CF; mubin++) finepairs[0].add_raw(bin, mubin, w_prod); 
}
for (int bin = 0; bin < NBIN_LONG; bin++) {
Float rmin_bin = rmin_long + bin * (rmax_long - rmin_long) / NBIN_LONG;
Float rmax_bin = rmin_long + (bin + 1) * (rmax_long - rmin_long) / NBIN_LONG;
Float w_prod = 0.5 * grid->sumw_neg * (2 * grid->sumw_pos + grid->sumw_neg) / pow(grid->max_boxsize, 3) * 4 * M_PI / 3 * (pow(rmax_bin, 3) - pow(rmin_bin, 3)); 
npcf[0].add_2pcf_long(bin, w_prod); 
}
#endif

powertime.Stop();

#ifndef OPENMP
#ifdef AVX
printf(
"\n# Time to compute required pairs (with AVX): %.2f\n\n",
accpairs.Elapsed());
#else
printf(
"\n# Time to compute required pairs (no AVX): %.2f\n\n",
accpairs.Elapsed());
#endif
#endif

printf("# We counted  %lld pairs within [%f %f].\n", cnt, rmin_short, rmax_short);
printf("# Average of %f pairs per primary particle.\n",
(Float)cnt / grid->np);
Float3 boxsize = grid->rect_boxsize;
float expected = grid->np * (4 * M_PI / 3.0) *
(pow(rmax_short, 3.0) - pow(rmin_short, 3.0)) /
(boxsize.x * boxsize.y * boxsize.z);
printf(
"# We expected %1.0f pairs per primary particle, off by a factor of "
"%f.\n",
expected, cnt / (expected * grid->np));

printf("# We counted  %lld triplets within [%f %f].\n", cnt3, rmin_short, rmax_short);

printf("# We counted  %lld pairs within [%f %f].\n", cnt2, rmin_long, rmax_long);
printf("# Average of %f pairs per primary particle.\n",
(Float)cnt2 / grid->np);
expected = grid->np * (4 * M_PI / 3.0) * (pow(rmax_long, 3.0) - pow(rmin_long, 3.0)) / (boxsize.x * boxsize.y * boxsize.z) / 2;
printf(
"# We expected %1.0f pairs per primary particle, off by a factor of "
"%f.\n",
expected, cnt2 / (expected * grid->np));

printf("\n# Accumulate Pairs: %6.3f s\n", accpairs.Elapsed());
printf("# Compute Power: %6.3f s\n\n", powertime.Elapsed());

delete[] pairs_i;

return;
}

#endif
