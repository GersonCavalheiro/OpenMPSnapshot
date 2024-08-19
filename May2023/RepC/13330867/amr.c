#include <signal.h>
#include <sys/types.h>
#include <par-res-kern_general.h>
#include <par-res-kern_fenix.h>
#include <random_draw.h>
#include <unistd.h>
#if DOUBLE
#define DTYPE     double
#define MPI_DTYPE MPI_DOUBLE
#define EPSILON   1.e-8
#define COEFX     1.0
#define COEFY     1.0
#define FSTR      "%10.4lf"
#else
#define DTYPE     float
#define MPI_DTYPE MPI_FLOAT
#define EPSILON   0.0001f
#define COEFX     1.0f
#define COEFY     1.0f
#define FSTR      "%10.4f"
#endif
#define INDEXIN(i,j)     (i+RADIUS+(j+RADIUS)*(L_width_bg+2*RADIUS))
#define IN(i,j)          in_bg[INDEXIN(i-L_istart_bg,j-L_jstart_bg)]
#define INDEXIN_R(g,i,j) (i+RADIUS+(j+RADIUS)*(L_width_r_true_gross[g]+2*RADIUS))
#define INDEXIN_RG(i,j)  (i+RADIUS+(j+RADIUS)*(L_width_r_true_gross+2*RADIUS))
#define IN_R(g,i,j)      in_r[g][INDEXIN_R(g,i-L_istart_r_true_gross[g],j-L_jstart_r_true_gross[g])]
#define ING_R(i,j)       ing_r[INDEXIN_RG(i-L_istart_r_true_gross,j-L_jstart_r_true_gross)]
#define INDEXOUT(i,j)    (i+(j)*(L_width_bg))
#define OUT(i,j)         out_bg[INDEXOUT(i-L_istart_bg,j-L_jstart_bg)]
#define INDEXOUT_R(i,j)  (i+(j)*L_width_r_true_gross[g])
#define OUT_R(g,i,j)     out_r[g][INDEXOUT_R(i-L_istart_r_true_gross[g],j-L_jstart_r_true_gross[g])]
#define WEIGHT(ii,jj)    weight[ii+RADIUS][jj+RADIUS]
#define WEIGHT_R(ii,jj)  weight_r[ii+RADIUS][jj+RADIUS]
#define undefined        1111
#define fine_grain       9797
#define no_talk          1212
#define high_water       3232
void time_step(int    Num_procs,
int    Num_procs_bg,
int    Num_procs_bgx, int Num_procs_bgy,
int    Num_procs_r[4],
int    Num_procs_rx[4], int Num_procs_ry[4],
int    my_ID,
int    my_ID_bg,
int    my_ID_bgx, int my_ID_bgy,
int    my_ID_r[4],
int    my_ID_rx[4], int my_ID_ry[4],
int    right_nbr_bg,
int    left_nbr_bg,
int    top_nbr_bg,
int    bottom_nbr_bg,
int    right_nbr_r[4],
int    left_nbr_r[4],
int    top_nbr_r[4],
int    bottom_nbr_r[4],
DTYPE  *top_buf_out_bg,
DTYPE  *top_buf_in_bg,
DTYPE  *bottom_buf_out_bg,
DTYPE  *bottom_buf_in_bg,
DTYPE  *right_buf_out_bg,
DTYPE  *right_buf_in_bg,
DTYPE  *left_buf_out_bg,
DTYPE  *left_buf_in_bg,
DTYPE  *top_buf_out_r[4],
DTYPE  *top_buf_in_r[4],
DTYPE  *bottom_buf_out_r[4],
DTYPE  *bottom_buf_in_r[4],
DTYPE  *right_buf_out_r[4],
DTYPE  *right_buf_in_r[4],
DTYPE  *left_buf_out_r[4],
DTYPE  *left_buf_in_r[4],
long   n,
int    refine_level,
long   G_istart_r[4],
long   G_iend_r[4],
long   G_jstart_r[4],
long   G_jend_r[4],
long   L_istart_bg, long L_iend_bg,
long   L_jstart_bg, long L_jend_bg,
long   L_width_bg, long L_height_bg,
long   L_istart_r[4], long L_iend_r[4],
long   L_jstart_r[4], long L_jend_r[4],
long   L_istart_r_gross[4], long L_iend_r_gross[4],
long   L_jstart_r_gross[4], long L_jend_r_gross[4],
long   L_istart_r_true_gross[4], long L_iend_r_true_gross[4],
long   L_jstart_r_true_gross[4], long L_jend_r_true_gross[4],
long   L_istart_r_true[4], long L_iend_r_true[4],
long   L_jstart_r_true[4], long L_jend_r_true[4],
long   L_width_r[4], long L_height_r[4],
long   L_width_r_true_gross[4], long L_height_r_true_gross[4], 
long   L_width_r_true[4], long L_height_r_true[4],
long   n_r,
long   n_r_true,
long   expand,
int    period,
int    duration,
int    sub_iterations, 
int    iter,
DTYPE  h_r,
int    num_interpolations,
DTYPE  * RESTRICT in_bg,
DTYPE  * RESTRICT out_bg,
DTYPE  * RESTRICT in_r[4],
DTYPE  * RESTRICT out_r[4],
DTYPE  weight[2*RADIUS+1][2*RADIUS+1],
DTYPE  weight_r[2*RADIUS+1][2*RADIUS+1],
int    load_balance,
MPI_Request request_bg[8],
MPI_Request request_r[4][8],
MPI_Comm comm_r[4],
MPI_Comm comm_bg,
int    first_through);
void get_BG_data(int load_balance, DTYPE *in_bg, DTYPE *ing_r, int my_ID, long expand,
int Num_procs, long L_width_bg, 
long L_istart_bg, long L_iend_bg, long L_jstart_bg, long L_jend_bg,
long L_istart_r, long L_iend_r, long L_jstart_r, long L_jend_r,
long G_istart_r, long G_jstart_r, MPI_Comm comm_bg, MPI_Comm comm_r,
long L_istart_r_gross, long L_iend_r_gross, 
long L_jstart_r_gross, long L_jend_r_gross, 
long L_width_r_true_gross, long L_istart_r_true_gross, long L_iend_r_true_gross,
long L_jstart_r_true_gross, long L_jend_r_true_gross, int g);
void interpolate(DTYPE *ing_r, long L_width_r_true_gross,
long L_istart_r_true_gross, long L_iend_r_true_gross,
long L_jstart_r_true_gross, long L_jend_r_true_gross, 
long L_istart_r_true, long L_iend_r_true,
long L_jstart_r_true, long L_jend_r_true, 
long expand, DTYPE h_r, int g, int Num_procs, int my_ID) {
long ir, jr, ib, jrb, jrb1, jb;
DTYPE xr, xb, yr, yb;
if (expand==1) return; 
for (jr=L_jstart_r_true_gross; jr<=L_jend_r_true_gross; jr+=expand) {
for (ir=L_istart_r_true_gross; ir<L_iend_r_true_gross; ir++) {
xr = h_r*(DTYPE)ir;
ib = (long)xr;
xb = (DTYPE)ib;
ING_R(ir,jr) = ING_R((ib+1)*expand,jr)*(xr-xb) +
ING_R(ib*expand,jr)*(xb+(DTYPE)1.0-xr);
}
}
for (jr=L_jstart_r_true; jr<=L_jend_r_true; jr++) {
yr = h_r*(DTYPE)jr;
jb = (long)yr;
jrb = jb*expand;
jrb1 = (jb+1)*expand;
yb = (DTYPE)jb;
for (ir=L_istart_r_true; ir<=L_iend_r_true; ir++) {
ING_R(ir,jr) = ING_R(ir,jrb1)*(yr-yb) + ING_R(ir,jrb)*(yb+(DTYPE)1.0-yr);
}
}
}
int main(int argc, char ** argv) {
int    Num_procs;         
int    Num_procs_bg;      
int    Num_procs_bgx, Num_procs_bgy; 
int    Num_procs_r[4];    
int    Num_procs_rx[4], Num_procs_ry[4];
int    my_ID;             
int    my_ID_bg;          
int    my_ID_bgx, my_ID_bgy;
int    my_ID_r[4];        
int    my_ID_rx[4], my_ID_ry[4];
int    right_nbr_bg;      
int    left_nbr_bg;       
int    top_nbr_bg;        
int    bottom_nbr_bg;     
int    right_nbr_r[4];    
int    left_nbr_r[4];     
int    top_nbr_r[4];      
int    bottom_nbr_r[4];   
DTYPE  *top_buf_out_bg;   
DTYPE  *top_buf_in_bg;    
DTYPE  *bottom_buf_out_bg;
DTYPE  *bottom_buf_in_bg; 
DTYPE  *right_buf_out_bg; 
DTYPE  *right_buf_in_bg;  
DTYPE  *left_buf_out_bg;  
DTYPE  *left_buf_in_bg;   
DTYPE  *top_buf_out_r[4]; 
DTYPE  *top_buf_in_r[4];  
DTYPE  *bottom_buf_out_r[4];
DTYPE  *bottom_buf_in_r[4];
DTYPE  *right_buf_out_r[4];
DTYPE  *right_buf_in_r[4];
DTYPE  *left_buf_out_r[4];
DTYPE  *left_buf_in_r[4]; 
int    root = 0;
long   n;                 
int    refine_level;      
long   G_istart_r[4];     
long   G_iend_r[4];       
long   G_jstart_r[4];     
long   G_jend_r[4];       
long   L_istart_bg, L_iend_bg;
long   L_jstart_bg, L_jend_bg;
long   L_width_bg, L_height_bg;
long   L_istart_r[4], L_iend_r[4];
long   L_jstart_r[4], L_jend_r[4];
long   L_istart_r_gross[4], L_iend_r_gross[4]; 
long   L_jstart_r_gross[4], L_jend_r_gross[4]; 
long   L_istart_r_true_gross[4], L_iend_r_true_gross[4]; 
long   L_jstart_r_true_gross[4], L_jend_r_true_gross[4]; 
long   L_istart_r_true[4], L_iend_r_true[4]; 
long   L_jstart_r_true[4], L_jend_r_true[4]; 
long   L_width_r[4], L_height_r[4]; 
long   L_width_r_true_gross[4], L_height_r_true_gross[4];
long   L_width_r_true[4], L_height_r_true[4];
int    g, g_loc;          
long   n_r;               
long   n_r_true;          
long   expand;            
int    period;            
int    duration;          
int    sub_iterations;    
long   i, j, ii, jj, it, jt, l, leftover; 
int    iter, iter_init, sub_iter;
DTYPE  norm, local_norm,  
reference_norm;
DTYPE  norm_in,           
local_norm_in,
reference_norm_in;
DTYPE  norm_r[4],         
local_norm_r[4],
reference_norm_r[4];
DTYPE  norm_in_r[4],      
local_norm_in_r[4],
reference_norm_in_r[4];
DTYPE  h_r;               
DTYPE  f_active_points_bg;
DTYPE  f_active_points_r; 
DTYPE  flops;             
int    iterations;        
int    iterations_r[4];   
int    full_cycles;       
int    leftover_iterations;
int    num_interpolations;
int    bg_updates;        
int    r_updates;          
double stencil_time,      
avgtime;
int    stencil_size;      
DTYPE  * RESTRICT in_bg;  
DTYPE  * RESTRICT out_bg; 
DTYPE  * RESTRICT in_r[4];
DTYPE  * RESTRICT out_r[4];
long   total_length_in;   
long   total_length_out;  
long   total_length_in_r[4]; 
long   total_length_out_r[4];
DTYPE  weight[2*RADIUS+1][2*RADIUS+1]; 
DTYPE  weight_r[2*RADIUS+1][2*RADIUS+1]; 
int    error=0;           
int    validate=1;        
char   *c_load_balance;   
int    load_balance;      
MPI_Request request_bg[8];
MPI_Request request_r[4][8];
MPI_Comm comm_r[4];       
MPI_Comm comm_bg;         
int    color_r;           
int    color_bg;          
int    rank_spread;       
int    spare_ranks;       
int    kill_ranks;        
int    *kill_set;         
int    kill_period;       
int    *fail_iter;        
int    fail_iter_s=0;     
DTYPE  init_add;          
int    checkpointing;     
int    num_fenix_init=1;  
int    num_fenix_init_loc;
int    num_failures;      
int    fenix_status;
random_draw_t dice;
int    first_through;     
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);
if (my_ID == root) {
printf("Parallel Research Kernels Version %s\n", PRKVERSION);
printf("MPI AMR stencil execution on 2D grid with Fenix fault tolerance\n");
#if !STAR
printf("ERROR: Compact stencil not supported\n");
error = 1;
goto ENDOFINPUTTESTS;
#endif
if (argc != 13 && argc != 14){
printf("Usage: %s <# iterations> <background grid size> <refinement size>\n",
*argv);
printf("       <refinement level> <refinement period> <refinement duration>\n");
printf("       <refinement sub-iterations> <spare ranks> <kill set size>\n");
printf("       <kill period> <checkpointing> <load balancer>\n");
printf("       load balancer: FINE_GRAIN [refinement rank spread]\n");
printf("                      NO_TALK\n");
printf("                      HIGH_WATER\n");
error = 1;
goto ENDOFINPUTTESTS;
}
iterations  = atoi(argv[1]); 
if (iterations < 1){
printf("ERROR: iterations must be >= 1 : %d \n",iterations);
error = 1;
goto ENDOFINPUTTESTS;
}
n  = atol(argv[2]);
if (n < 2){
printf("ERROR: grid must have at least one cell: %ld\n", n);
error = 1;
goto ENDOFINPUTTESTS;
}
n_r = atol(argv[3]);
if (n_r < 2) {
printf("ERROR: refinements must have at least one cell: %ld\n", n_r);
error = 1;
goto ENDOFINPUTTESTS;
}
if (n_r>n) {
printf("ERROR: refinements must be contained in background grid: %ld\n", n_r);
error = 1;
goto ENDOFINPUTTESTS;
}
refine_level = atoi(argv[4]);
if (refine_level < 0) {
printf("ERROR: refinement levels must be >= 0 : %d\n", refine_level);
error = 1;
goto ENDOFINPUTTESTS;
}
period = atoi(argv[5]);
if (period < 1) {
printf("ERROR: refinement period must be at least one: %d\n", period);
error = 1;
goto ENDOFINPUTTESTS;
}
duration = atoi(argv[6]);
if (duration < 1 || duration > period) {
printf("ERROR: refinement duration must be positive, no greater than period: %d\n",
duration);
error = 1;
goto ENDOFINPUTTESTS;
}
sub_iterations = atoi(argv[7]);
if (sub_iterations < 1) {
printf("ERROR: refinement sub-iterations must be positive: %d\n", sub_iterations);
error = 1;
goto ENDOFINPUTTESTS;
}
spare_ranks  = atoi(argv[8]);
if (spare_ranks < 0 || spare_ranks >= Num_procs){
printf("ERROR: Illegal number of spare ranks : %d \n", spare_ranks);
error = 1;
goto ENDOFINPUTTESTS;     
}
kill_ranks = atoi(argv[9]);
if (kill_ranks < 0 || kill_ranks > spare_ranks) {
printf("ERROR: Number of ranks in kill set invalid: %d\n", kill_ranks);
error = 1;
goto ENDOFINPUTTESTS;     
}
kill_period = atoi(argv[10]);
if (kill_period < 1) {
printf("ERROR: rank kill period must be positive: %d\n", kill_period);
error = 1;
goto ENDOFINPUTTESTS;     
}
checkpointing = atoi(argv[11]);
if (checkpointing) {
printf("ERROR: Fenix checkpointing not yet implemented\n");
error = 1;
goto ENDOFINPUTTESTS;     
}
c_load_balance = argv[12];
if      (!strcmp("FINE_GRAIN", c_load_balance)) load_balance=fine_grain;
else if (!strcmp("NO_TALK",    c_load_balance)) load_balance=no_talk;
else if (!strcmp("HIGH_WATER", c_load_balance)) load_balance=high_water;
else                                            load_balance=undefined;
if (load_balance==undefined) {
printf("ERROR: invalid load balancer %s\n", c_load_balance);
error = 1;
goto ENDOFINPUTTESTS;
}
if (load_balance == high_water && Num_procs==1) {
printf("ERROR: Load balancer HIGH_WATER requires more than one rank\n");
error = 1;
goto ENDOFINPUTTESTS;
}
if (load_balance==fine_grain && argc==14) {
rank_spread = atoi(argv[13]);
if (rank_spread<1 || rank_spread>Num_procs) {
printf("ERROR: Invalid number of ranks to spread refinement work: %d\n", rank_spread);
error = 1;
goto ENDOFINPUTTESTS;
}
} else rank_spread = Num_procs-spare_ranks;
if (RADIUS < 1) {
printf("ERROR: Stencil radius %d should be positive\n", RADIUS);
error = 1;
goto ENDOFINPUTTESTS;
}
if (2*RADIUS+1 > n) {
printf("ERROR: Stencil radius %d exceeds grid size %ld\n", RADIUS, n);
error = 1;
goto ENDOFINPUTTESTS;
}
h_r = (DTYPE)1.0; expand = 1;
for (l=0; l<refine_level; l++) {
h_r /= (DTYPE)2.0;
expand *= 2;
}
n_r_true = (n_r-1)*expand+1;
if (2*RADIUS+1 > n_r_true) {
printf("ERROR: Stencil radius %d exceeds refinement size %ld\n", RADIUS, n_r_true);
error = 1;
goto ENDOFINPUTTESTS;
}
ENDOFINPUTTESTS:;  
}
bail_out(error);
MPI_Bcast(&n,              1, MPI_LONG,  root, MPI_COMM_WORLD);
MPI_Bcast(&n_r,            1, MPI_LONG,  root, MPI_COMM_WORLD);
MPI_Bcast(&h_r,            1, MPI_DTYPE, root, MPI_COMM_WORLD);
MPI_Bcast(&n_r_true,       1, MPI_LONG,  root, MPI_COMM_WORLD);
MPI_Bcast(&period,         1, MPI_INT,   root, MPI_COMM_WORLD);
MPI_Bcast(&duration,       1, MPI_INT,   root, MPI_COMM_WORLD);
MPI_Bcast(&refine_level,   1, MPI_INT,   root, MPI_COMM_WORLD);
MPI_Bcast(&iterations,     1, MPI_INT,   root, MPI_COMM_WORLD);
MPI_Bcast(&sub_iterations, 1, MPI_INT,   root, MPI_COMM_WORLD);
MPI_Bcast(&load_balance,   1, MPI_INT,   root, MPI_COMM_WORLD);
MPI_Bcast(&rank_spread,    1, MPI_INT,   root, MPI_COMM_WORLD);
MPI_Bcast(&expand,         1, MPI_LONG,  root, MPI_COMM_WORLD);
MPI_Bcast(&spare_ranks,    1, MPI_INT,   root, MPI_COMM_WORLD);
MPI_Bcast(&kill_ranks,     1, MPI_INT,   root, MPI_COMM_WORLD);
MPI_Bcast(&kill_period,    1, MPI_INT,   root, MPI_COMM_WORLD);
MPI_Bcast(&checkpointing,  1, MPI_INT,   root, MPI_COMM_WORLD);
LCG_init(&dice);
for (iter=0; iter<=iterations; iter++) {
fail_iter_s += random_draw(kill_period, &dice);
if (fail_iter_s > iterations) break;
num_fenix_init++;
}
if ((num_fenix_init-1)*kill_ranks>spare_ranks) {
if (my_ID==0) printf("ERROR: number of injected errors %d exceeds spare ranks %d\n",
(num_fenix_init-1)*kill_ranks, spare_ranks);
error = 1;
}
else num_failures = num_fenix_init-1;
bail_out(error);
if ((num_fenix_init-1)*kill_ranks>=Num_procs-spare_ranks) if (my_ID==root)
printf("WARNING: All active ranks will be replaced by recovered ranks; timings not valid\n");
fail_iter = (int *) prk_malloc(sizeof(int)*num_fenix_init);
if (!fail_iter) {
printf("ERROR: Rank %d could not allocate space for array fail_iter\n", my_ID);
error = 1;
}
bail_out(error);
LCG_init(&dice);
for (fail_iter_s=iter=0; iter<num_fenix_init; iter++) {
fail_iter_s += random_draw(kill_period, &dice);
fail_iter[iter] = fail_iter_s;
}
Fenix_Init(&fenix_status, MPI_COMM_WORLD, NULL, &argc, &argv, spare_ranks, 
0, MPI_INFO_NULL, &error);
if (error==FENIX_WARNING_SPARE_RANKS_DEPLETED) 
printf("ERROR: Rank %d: Cannot reconstitute original communicator\n", my_ID);
bail_out(error);
MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);
switch (fenix_status){
case FENIX_ROLE_INITIAL_RANK:   
first_through = iter_init = 0;
num_fenix_init_loc =  0;    
break;
case FENIX_ROLE_RECOVERED_RANK: 
first_through = 1; 
iter_init     = iterations + 1;
num_fenix_init_loc = iterations + 1;
break;
case FENIX_ROLE_SURVIVOR_RANK:  
first_through = 1; 
iter_init = iter; 
num_fenix_init_loc++;
}
MPI_Allreduce(&iter_init, &iter, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
MPI_Allreduce(&num_fenix_init_loc, &num_fenix_init, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
float bg_size, total_size, Frac_procs_bg; 
switch (load_balance) {
case fine_grain: MPI_Comm_dup(MPI_COMM_WORLD, &comm_bg);
Num_procs_bg = Num_procs;
my_ID_bg = my_ID;
for (g=0; g<4; g++) {
if (my_ID < rank_spread) color_r = 1;
else                     color_r = MPI_UNDEFINED;
MPI_Comm_split(MPI_COMM_WORLD, color_r, my_ID, &comm_r[g]);
if (comm_r[g] != MPI_COMM_NULL) {
MPI_Comm_size(comm_r[g], &Num_procs_r[g]);
MPI_Comm_rank(comm_r[g], &my_ID_r[g]);
}
}
break;
case no_talk:    MPI_Comm_dup(MPI_COMM_WORLD, &comm_bg);
Num_procs_bg = Num_procs;
my_ID_bg = my_ID;
break;
case high_water: bg_size=n*n; 
total_size = n*n+n_r_true*n_r_true;
Frac_procs_bg;
Frac_procs_bg = (float) Num_procs * bg_size/total_size;
Num_procs_bg = MIN(Num_procs-1,MAX(1,ceil(Frac_procs_bg)));
int Num_procs_R = Num_procs-Num_procs_bg;
optimize_split(&Num_procs_bg, &Num_procs_R, 3);
if (my_ID>=Num_procs_bg) {color_bg = MPI_UNDEFINED; color_r = 1;}
else                     {color_bg = 1; color_r = MPI_UNDEFINED;}
MPI_Comm_split(MPI_COMM_WORLD, color_bg, my_ID, &comm_bg);
if (comm_bg != MPI_COMM_NULL) {
MPI_Comm_size(comm_bg, &Num_procs_bg);
MPI_Comm_rank(comm_bg, &my_ID_bg);
}
for (g=0; g<4; g++) {
MPI_Comm_split(MPI_COMM_WORLD, color_r, my_ID, &comm_r[g]);
if (comm_r[g] != MPI_COMM_NULL) {
MPI_Comm_size(comm_r[g], &Num_procs_r[g]);
MPI_Comm_rank(comm_r[g], &my_ID_r[g]);
} 
else {
Num_procs_r[g] = Num_procs - Num_procs_bg;
}
}
if (comm_bg == MPI_COMM_NULL) Num_procs_bg = Num_procs - Num_procs_r[0];
break;
}
if (comm_bg != MPI_COMM_NULL) {
factor(Num_procs_bg, &Num_procs_bgx, &Num_procs_bgy);
my_ID_bgx = my_ID_bg%Num_procs_bgx;
my_ID_bgy = my_ID_bg/Num_procs_bgx;
right_nbr_bg = left_nbr_bg = top_nbr_bg = bottom_nbr_bg = -1;
if (my_ID_bgx < Num_procs_bgx-1) right_nbr_bg  = my_ID+1;
if (my_ID_bgx > 0)               left_nbr_bg   = my_ID-1;
if (my_ID_bgy < Num_procs_bgy-1) top_nbr_bg    = my_ID+Num_procs_bgx;
if (my_ID_bgy > 0)               bottom_nbr_bg = my_ID-Num_procs_bgx;
L_width_bg = n/Num_procs_bgx;
leftover = n%Num_procs_bgx;
if (my_ID_bgx<leftover) {
L_istart_bg = (L_width_bg+1) * my_ID_bgx; 
L_iend_bg = L_istart_bg + L_width_bg;
}
else {
L_istart_bg = (L_width_bg+1) * leftover + L_width_bg * (my_ID_bgx-leftover);
L_iend_bg = L_istart_bg + L_width_bg - 1;
}
L_width_bg = L_iend_bg - L_istart_bg + 1;
if (L_width_bg == 0) {
printf("ERROR: rank %d has no work to do\n", my_ID);
error = 1;
goto ENDOFBG;
}
L_height_bg = n/Num_procs_bgy;
leftover = n%Num_procs_bgy;
if (my_ID_bgy<leftover) {
L_jstart_bg = (L_height_bg+1) * my_ID_bgy; 
L_jend_bg = L_jstart_bg + L_height_bg;
}
else {
L_jstart_bg = (L_height_bg+1) * leftover + L_height_bg * (my_ID_bgy-leftover);
L_jend_bg = L_jstart_bg + L_height_bg - 1;
}
L_height_bg = L_jend_bg - L_jstart_bg + 1;
if (L_height_bg == 0) {
printf("ERROR: rank %d has no work to do\n", my_ID);
error = 1;
goto ENDOFBG;
}
if (L_width_bg < RADIUS || L_height_bg < RADIUS) {
printf("ERROR: rank %d's BG work tile smaller than stencil radius: %d\n",
my_ID, MIN(L_width_bg, L_height_bg));
error = 1;
goto ENDOFBG;
}
total_length_in  = (long) (L_width_bg+2*RADIUS)*(long) (L_height_bg+2*RADIUS);
total_length_out = (long) L_width_bg* (long) L_height_bg;
if (fenix_status != FENIX_ROLE_SURVIVOR_RANK) {
in_bg  = (DTYPE *) prk_malloc(total_length_in*sizeof(DTYPE));
out_bg = (DTYPE *) prk_malloc(total_length_out*sizeof(DTYPE));
if (!in_bg || !out_bg) {
printf("ERROR: rank %d could not allocate space for input/output array\n",
my_ID);
error = 1;
goto ENDOFBG;
}
}
ENDOFBG:;
}
else { 
L_istart_bg =  0;
L_iend_bg   = -1;
L_jstart_bg =  0;;
L_jend_bg   = -1;
}
bail_out(error);
G_istart_r[0] = G_istart_r[2] = 0;
G_iend_r[0]   = G_iend_r[2]   = n_r-1;
G_istart_r[1] = G_istart_r[3] = n-n_r;
G_iend_r[1]   = G_iend_r[3]   = n-1;
G_jstart_r[0] = G_jstart_r[3] = 0;
G_jend_r[0]   = G_jend_r[3]   = n_r-1;
G_jstart_r[1] = G_jstart_r[2] = n-n_r;
G_jend_r[1]   = G_jend_r[2]   = n-1;
switch(load_balance) {
case no_talk:    
for (g=0; g<4; g++) {
L_istart_r[g] = MAX(L_istart_bg,G_istart_r[g]);
L_iend_r[g]   = MIN(L_iend_bg,  G_iend_r[g]);		     
L_jstart_r[g] = MAX(L_jstart_bg,G_jstart_r[g]);
L_jend_r[g]   = MIN(L_jend_bg,  G_jend_r[g]);
if (L_istart_r[g]<=L_iend_r[g] &&
L_jstart_r[g]<=L_jend_r[g]) color_r = 1;
else                            color_r = MPI_UNDEFINED;
MPI_Comm_split(MPI_COMM_WORLD, color_r, my_ID, &comm_r[g]);
if (comm_r[g] != MPI_COMM_NULL) {
MPI_Comm_size(comm_r[g], &Num_procs_r[g]);
MPI_Comm_rank(comm_r[g], &my_ID_r[g]);
long ilow, ihigh, jlow, jhigh;
MPI_Allreduce(&my_ID_bgx,&ilow ,1,MPI_LONG,MPI_MIN,comm_r[g]);
MPI_Allreduce(&my_ID_bgx,&ihigh,1,MPI_LONG,MPI_MAX,comm_r[g]);
MPI_Allreduce(&my_ID_bgy,&jlow ,1,MPI_LONG,MPI_MIN,comm_r[g]);
MPI_Allreduce(&my_ID_bgy,&jhigh,1,MPI_LONG,MPI_MAX,comm_r[g]);
Num_procs_rx[g] = ihigh-ilow+1;
Num_procs_ry[g] = jhigh-jlow+1;
}
}
break;
case fine_grain: 
case high_water: 
for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL) {
factor(Num_procs_r[g], &Num_procs_rx[g], &Num_procs_ry[g]);
}
break;
}
for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL) {
my_ID_rx[g] = my_ID_r[g]%Num_procs_rx[g];
my_ID_ry[g] = my_ID_r[g]/Num_procs_rx[g];
right_nbr_r[g] = left_nbr_r[g] = top_nbr_r[g] = bottom_nbr_r[g] = -1;
if (my_ID_rx[g] < Num_procs_rx[g]-1) right_nbr_r[g]  = my_ID_r[g]+1;
if (my_ID_rx[g] > 0)                 left_nbr_r[g]   = my_ID_r[g]-1;
if (my_ID_ry[g] < Num_procs_ry[g]-1) top_nbr_r[g]    = my_ID_r[g]+Num_procs_rx[g];
if (my_ID_ry[g] > 0)                 bottom_nbr_r[g] = my_ID_r[g]-Num_procs_rx[g];
}
MPI_Barrier(MPI_COMM_WORLD);
if (my_ID == root && fenix_status == FENIX_ROLE_INITIAL_RANK) {
printf("Number of ranks                 = %d\n", Num_procs + spare_ranks);
printf("Background grid size            = %ld\n", n);
printf("Radius of stencil               = %d\n", RADIUS);
printf("Tiles in x/y-direction on BG    = %d/%d\n", Num_procs_bgx, Num_procs_bgy);
}
for (g=0; g<4; g++) {
MPI_Barrier(MPI_COMM_WORLD);
if ((comm_r[g] != MPI_COMM_NULL) && (my_ID_r[g]==root) &&
fenix_status == FENIX_ROLE_INITIAL_RANK)
printf("Tiles in x/y-direction on ref %d = %d/%d\n",
g, Num_procs_rx[g], Num_procs_ry[g]);
prk_pause(0.001); 
}
MPI_Barrier(MPI_COMM_WORLD);
if (my_ID == root && fenix_status == FENIX_ROLE_INITIAL_RANK) {
printf("Type of stencil                 = star\n");
#if DOUBLE
printf("Data type                       = double precision\n");
#else
printf("Data type                       = single precision\n");
#endif
#if LOOPGEN
printf("Script used to expand stencil loop body\n");
#else
printf("Compact representation of stencil loop body\n");
#endif
printf("Number of iterations            = %d\n", iterations);
printf("Load balancer                   = %s\n", c_load_balance);
if (load_balance==fine_grain)
printf("Refinement rank spread          = %d\n", rank_spread);
printf("Refinements:\n");
printf("   Background grid points       = %ld\n", n_r);
printf("   Grid size                    = %ld\n", n_r_true);
printf("   Refinement level             = %d\n", refine_level);
printf("   Period                       = %d\n", period);
printf("   Duration                     = %d\n", duration);
printf("   Sub-iterations               = %d\n", sub_iterations);
printf("Number of spare ranks           = %d\n", spare_ranks);
printf("Kill set size                   = %d\n", kill_ranks);
printf("Fault period                    = %d\n", kill_period);
printf("Total injected failures         = %d times %d errors\n", 
num_failures, kill_ranks);
if (checkpointing)
printf("Data recovery                   = Fenix checkpointing\n");
else
printf("Data recovery                   = analytical\n");
}
for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL) {
if (load_balance==fine_grain || load_balance==high_water) {
L_width_r[g] = n_r/Num_procs_rx[g];
leftover =   n_r%Num_procs_rx[g];
if (my_ID_rx[g]<leftover) {
L_istart_r[g] = (L_width_r[g]+1) * my_ID_rx[g]; 
L_iend_r[g]   = L_istart_r[g] + L_width_r[g];
}
else {
L_istart_r[g] = (L_width_r[g]+1) * leftover + L_width_r[g] * (my_ID_rx[g]-leftover);
L_iend_r[g]   = L_istart_r[g] + L_width_r[g] - 1;
}
L_height_r[g] = n_r/Num_procs_ry[g];
leftover = n_r%Num_procs_ry[g];
if (my_ID_ry[g]<leftover) {
L_jstart_r[g] = (L_height_r[g]+1) * my_ID_ry[g]; 
L_jend_r[g]   = L_jstart_r[g] + L_height_r[g];
}
else {
L_jstart_r[g] = (L_height_r[g]+1) * leftover + L_height_r[g] * (my_ID_ry[g]-leftover);
L_jend_r[g]   = L_jstart_r[g] + L_height_r[g] - 1;
}
L_width_r_true[g] = n_r_true/Num_procs_rx[g];
leftover =   n_r_true%Num_procs_rx[g];
if (my_ID_rx[g]<leftover) {
L_istart_r_true[g] = (L_width_r_true[g]+1) * my_ID_rx[g]; 
L_iend_r_true[g]   = L_istart_r_true[g] + L_width_r_true[g];
}
else {
L_istart_r_true[g] = (L_width_r_true[g]+1) * leftover + L_width_r_true[g] * (my_ID_rx[g]-leftover);
L_iend_r_true[g]   = L_istart_r_true[g] + L_width_r_true[g] - 1;
}
L_height_r_true[g] = n_r_true/Num_procs_ry[g];
leftover = n_r_true%Num_procs_ry[g];
if (my_ID_ry[g]<leftover) {
L_jstart_r_true[g] = (L_height_r_true[g]+1) * my_ID_ry[g]; 
L_jend_r_true[g]   = L_jstart_r_true[g] + L_height_r_true[g];
}
else {
L_jstart_r_true[g] = (L_height_r_true[g]+1) * leftover + L_height_r_true[g] * (my_ID_ry[g]-leftover);
L_jend_r_true[g]   = L_jstart_r_true[g] + L_height_r_true[g] - 1;
}
L_istart_r[g] += G_istart_r[g]; L_iend_r[g] += G_istart_r[g];
L_jstart_r[g] += G_jstart_r[g]; L_jend_r[g] += G_jstart_r[g];
}
else if (load_balance == no_talk) { 
L_istart_r_true[g] = (L_istart_r[g] - G_istart_r[g])*expand;
if (my_ID_rx[g]>0) L_istart_r_true[g] -= expand/2;
L_iend_r_true[g]   = (L_iend_r[g]   - G_istart_r[g])*expand;
if (my_ID_rx[g] < Num_procs_rx[g]-1) L_iend_r_true[g] += (expand-1)/2;
L_jstart_r_true[g] = (L_jstart_r[g] - G_jstart_r[g])*expand;
if (my_ID_ry[g]>0) L_jstart_r_true[g] -= expand/2;
L_jend_r_true[g]   = (L_jend_r[g]   - G_jstart_r[g])*expand;
if (my_ID_ry[g] < Num_procs_ry[g]-1) L_jend_r_true[g] += (expand-1)/2;
}
L_istart_r_true_gross[g] = (L_istart_r_true[g]/expand)*expand;
L_iend_r_true_gross[g]   = (L_iend_r_true[g]/expand+1)*expand;
L_jstart_r_true_gross[g] = (L_jstart_r_true[g]/expand)*expand;
L_jend_r_true_gross[g]   = (L_jend_r_true[g]/expand+1)*expand;
L_istart_r_gross[g]      = L_istart_r_true_gross[g]/expand;
L_iend_r_gross[g]        = L_iend_r_true_gross[g]/expand;
L_jstart_r_gross[g]      = L_jstart_r_true_gross[g]/expand;
L_jend_r_gross[g]        = L_jend_r_true_gross[g]/expand;
L_istart_r_gross[g] += G_istart_r[g]; L_iend_r_gross[g] += G_istart_r[g];
L_jstart_r_gross[g] += G_jstart_r[g]; L_jend_r_gross[g] += G_jstart_r[g];
L_height_r[g]            = L_jend_r[g] -            L_jstart_r[g] + 1;
L_width_r[g]             = L_iend_r[g] -            L_istart_r[g] + 1;
L_height_r_true_gross[g] = L_jend_r_true_gross[g] - L_jstart_r_true_gross[g] + 1;
L_width_r_true_gross[g]  = L_iend_r_true_gross[g] - L_istart_r_true_gross[g] + 1;
L_height_r_true[g]       = L_jend_r_true[g] -       L_jstart_r_true[g] + 1;
L_width_r_true[g]        = L_iend_r_true[g] -       L_istart_r_true[g] + 1;
if (L_height_r_true[g] == 0 || L_width_r_true[g] == 0)  {
printf("ERROR: rank %d has no work to do on refinement %d\n", my_ID, g);
error = 1;
}
if (L_width_r_true[g] < RADIUS || L_height_r_true[g] < RADIUS) {
printf("ERROR: rank %d's work tile %d smaller than stencil radius: %d\n", 
my_ID, g, MIN(L_width_r_true[g],L_height_r_true[g]));
error = 1;
}
total_length_in_r[g]  = (L_width_r_true_gross[g]+2*RADIUS)*
(L_height_r_true_gross[g]+2*RADIUS);
total_length_out_r[g] = L_width_r_true_gross[g] * L_height_r_true_gross[g];
if (fenix_status != FENIX_ROLE_SURVIVOR_RANK) {
in_r[g]  = (DTYPE *) prk_malloc(sizeof(DTYPE)*total_length_in_r[g]);
out_r[g] = (DTYPE *) prk_malloc(sizeof(DTYPE)*total_length_out_r[g]);
if (!in_r[g] || !out_r[g]) {
printf("ERROR: could not allocate space for refinement input or output arrays\n");
error=1;
}
}
}
else {
L_istart_r_gross[g] =  0;
L_iend_r_gross[g]   = -1;
L_jstart_r_gross[g] =  0;
L_jend_r_gross[g]   = -1;
}
bail_out(error);
for (jj=-RADIUS; jj<=RADIUS; jj++) for (ii=-RADIUS; ii<=RADIUS; ii++) 
WEIGHT(ii,jj) = (DTYPE) 0.0;
stencil_size = 4*RADIUS+1;
for (ii=1; ii<=RADIUS; ii++) {
WEIGHT(0, ii) = WEIGHT( ii,0) =  (DTYPE) (1.0/(2.0*ii*RADIUS));
WEIGHT(0,-ii) = WEIGHT(-ii,0) = -(DTYPE) (1.0/(2.0*ii*RADIUS));
}
for (jj=-RADIUS; jj<=RADIUS; jj++) for (ii=-RADIUS; ii<=RADIUS; ii++)
WEIGHT_R(ii,jj) = WEIGHT(ii,jj)*(DTYPE)expand;
f_active_points_bg = (DTYPE) (n-2*RADIUS)*(DTYPE) (n-2*RADIUS);
f_active_points_r  = (DTYPE) (n_r_true-2*RADIUS)*(DTYPE) (n_r_true-2*RADIUS);
if (comm_bg != MPI_COMM_NULL)
if (checkpointing) init_add = 0.0;
else               init_add = (DTYPE) iter;
for (j=L_jstart_bg; j<=L_jend_bg; j++) for (i=L_istart_bg; i<=L_iend_bg; i++) {
IN(i,j)  = COEFX*i+COEFY*j+init_add;
OUT(i,j) = (COEFX+COEFY)*init_add;
}
if (comm_bg != MPI_COMM_NULL && fenix_status != FENIX_ROLE_SURVIVOR_RANK) {
top_buf_out_bg = (DTYPE *) prk_malloc(4*sizeof(DTYPE)*RADIUS*L_width_bg);
if (!top_buf_out_bg) {
printf("ERROR: Rank %d could not allocate comm buffers for y-direction\n", my_ID);
error = 1;
} 
top_buf_in_bg     = top_buf_out_bg +   RADIUS*L_width_bg;
bottom_buf_out_bg = top_buf_out_bg + 2*RADIUS*L_width_bg;
bottom_buf_in_bg  = top_buf_out_bg + 3*RADIUS*L_width_bg;
right_buf_out_bg  = (DTYPE *) prk_malloc(4*sizeof(DTYPE)*RADIUS*(L_height_bg+2));
if (!right_buf_out_bg) {
printf("ERROR: Rank %d could not allocate comm buffers for x-direction\n", my_ID);
error = 1;
}
right_buf_in_bg   = right_buf_out_bg +   RADIUS*(L_height_bg+2);
left_buf_out_bg   = right_buf_out_bg + 2*RADIUS*(L_height_bg+2);
left_buf_in_bg    = right_buf_out_bg + 3*RADIUS*(L_height_bg+2);
}
bail_out(error);
if (!checkpointing) {
full_cycles = iter/(period*4);
leftover_iterations = iter%(period*4);
for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL) {
iterations_r[g] = sub_iterations*(full_cycles*duration+
MIN(MAX(0,leftover_iterations-g*period),duration));
for (j=L_jstart_r_true[g]; j<=L_jend_r_true[g]; j++) 
for (i=L_istart_r_true[g]; i<=L_iend_r_true[g]; i++) {
OUT_R(g,i,j) = (DTYPE) iterations_r[g] * (COEFX + COEFY);
IN_R(g,i,j)  = (DTYPE)0.0;
}
}
}
else for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL) {
for (j=L_jstart_r_true[g]; j<=L_jend_r_true[g]; j++) 
for (i=L_istart_r_true[g]; i<=L_iend_r_true[g]; i++) {
IN_R(g,i,j)  = (DTYPE)0.0;
OUT_R(g,i,j) = (DTYPE)0.0;
}
}
for (g=0; g<4; g++) if (comm_r[g] != MPI_COMM_NULL &&
fenix_status != FENIX_ROLE_SURVIVOR_RANK) {
top_buf_out_r[g] = (DTYPE *) prk_malloc(4*sizeof(DTYPE)*RADIUS*L_width_r_true[g]);
if (!top_buf_out_r[g]) {
printf("ERROR: Rank %d could not allocate comm buffers for y-direction for r=%d\n", 
my_ID, g);
error = 1;
}
top_buf_in_r[g]     = top_buf_out_r[g] +   RADIUS*L_width_r_true[g];
bottom_buf_out_r[g] = top_buf_out_r[g] + 2*RADIUS*L_width_r_true[g];
bottom_buf_in_r[g]  = top_buf_out_r[g] + 3*RADIUS*L_width_r_true[g];
right_buf_out_r[g]  = (DTYPE *) prk_malloc(4*sizeof(DTYPE)*RADIUS*L_height_r_true[g]);
if (!right_buf_out_r[g]) {
printf("ERROR: Rank %d could not allocate comm buffers for x-direction for r=%d\n", my_ID, g);
error = 1;
}
right_buf_in_r[g]   = right_buf_out_r[g] +   RADIUS*L_height_r_true[g];
left_buf_out_r[g]   = right_buf_out_r[g] + 2*RADIUS*L_height_r_true[g];
left_buf_in_r[g]    = right_buf_out_r[g] + 3*RADIUS*L_height_r_true[g];
}
bail_out(error);
num_interpolations = 0;
for (; iter<=iterations; iter++){
if (iter == 1) {
MPI_Barrier(MPI_COMM_WORLD);
stencil_time = wtime();
}
if (iter == fail_iter[num_fenix_init]) {
pid_t pid = getpid();
if (my_ID < kill_ranks) {
#if VERBOSE
printf("Rank %d, pid %d commits suicide in iter %d\n", my_ID, pid, iter);
#endif
kill(pid, SIGKILL);
}
#if VERBOSE
else printf("Rank %d, pid %d is survivor rank in iter %d\n", my_ID, pid, iter);
#endif
}  
time_step(Num_procs, Num_procs_bg, Num_procs_bgx, Num_procs_bgy,
Num_procs_r, Num_procs_rx, Num_procs_ry,
my_ID, my_ID_bg, my_ID_bgx, my_ID_bgy, my_ID_r, my_ID_rx, my_ID_ry,
right_nbr_bg, left_nbr_bg, top_nbr_bg, bottom_nbr_bg,
right_nbr_r, left_nbr_r, top_nbr_r, bottom_nbr_r,
top_buf_out_bg, top_buf_in_bg, bottom_buf_out_bg, bottom_buf_in_bg,
right_buf_out_bg, right_buf_in_bg, left_buf_out_bg, left_buf_in_bg,
top_buf_out_r, top_buf_in_r, bottom_buf_out_r, bottom_buf_in_r,
right_buf_out_r, right_buf_in_r, left_buf_out_r, left_buf_in_r,
n, refine_level, G_istart_r, G_iend_r, G_jstart_r, G_jend_r,
L_istart_bg, L_iend_bg, L_jstart_bg, L_jend_bg, L_width_bg, L_height_bg,
L_istart_r, L_iend_r, L_jstart_r, L_jend_r,
L_istart_r_gross, L_iend_r_gross, L_jstart_r_gross, L_jend_r_gross,
L_istart_r_true_gross, L_iend_r_true_gross,
L_jstart_r_true_gross, L_jend_r_true_gross,
L_istart_r_true, L_iend_r_true, L_jstart_r_true, L_jend_r_true,
L_width_r, L_height_r, L_width_r_true_gross, L_height_r_true_gross, 
L_width_r_true, L_height_r_true,
n_r, n_r_true, expand, period, duration, sub_iterations, iter, h_r,
num_interpolations, in_bg, out_bg, in_r, out_r, weight, weight_r,
load_balance, request_bg, request_r, comm_r, comm_bg, first_through);
} 
MPI_Barrier(MPI_COMM_WORLD);
stencil_time = wtime() - stencil_time;
local_norm = (DTYPE) 0.0;
if (comm_bg != MPI_COMM_NULL) 
for (int j=MAX(L_jstart_bg,RADIUS); j<=MIN(n-RADIUS-1,L_jend_bg); j++) {
for (int i=MAX(L_istart_bg,RADIUS); i<=MIN(n-RADIUS-1,L_iend_bg); i++) {
local_norm += (DTYPE)ABS(OUT(i,j));
}
}
root = Num_procs-1;
MPI_Reduce(&local_norm, &norm, 1, MPI_DTYPE, MPI_SUM, root, MPI_COMM_WORLD);
if (my_ID == root) norm /= f_active_points_bg;
local_norm_in = (DTYPE) 0.0;
if (comm_bg != MPI_COMM_NULL) 
for (j=L_jstart_bg; j<=L_jend_bg; j++) for (i=L_istart_bg; i<=L_iend_bg; i++) {
local_norm_in += (DTYPE)ABS(IN(i,j));
}
MPI_Reduce(&local_norm_in, &norm_in, 1, MPI_DTYPE, MPI_SUM, root, MPI_COMM_WORLD);
if (my_ID == root) norm_in /= n*n;
for (g=0; g<4; g++) {
local_norm_r[g] = local_norm_in_r[g] = (DTYPE) 0.0;
if (comm_r[g] != MPI_COMM_NULL)
for (j=MAX(L_jstart_r_true[g],RADIUS); j<=MIN(n_r_true-RADIUS-1,L_jend_r_true[g]); j++) 
for (i=MAX(L_istart_r_true[g],RADIUS); i<=MIN(n_r_true-RADIUS-1,L_iend_r_true[g]); i++) {
local_norm_r[g] += (DTYPE)ABS(OUT_R(g,i,j));
}
MPI_Reduce(&local_norm_r[g], &norm_r[g], 1, MPI_DTYPE, MPI_SUM, root, MPI_COMM_WORLD);
if (my_ID == root) norm_r[g] /= f_active_points_r;
if (comm_r[g] != MPI_COMM_NULL)
for (j=L_jstart_r_true[g]; j<=L_jend_r_true[g]; j++) 
for (i=L_istart_r_true[g]; i<=L_iend_r_true[g]; i++) {
local_norm_in_r[g] += (DTYPE)ABS(IN_R(g,i,j)); 
}
MPI_Reduce(&local_norm_in_r[g], &norm_in_r[g], 1, MPI_DTYPE, MPI_SUM, root, MPI_COMM_WORLD);
if (my_ID == root) norm_in_r[g] /=  n_r_true*n_r_true;
}
if (my_ID == root) {
reference_norm = (DTYPE) (iterations+1) * (COEFX + COEFY);
reference_norm_in = (COEFX+COEFY)*(DTYPE)((n-1)/2.0)+iterations+1;
if (ABS(norm-reference_norm) > EPSILON) {
printf("ERROR: L1 norm = "FSTR", Reference L1 norm = "FSTR"\n",
norm, reference_norm);
validate = 0;
}
else {
#if VERBOSE
printf("SUCCESS: Reference L1 norm         = "FSTR", L1 norm         = "FSTR"\n", 
reference_norm, norm);
#endif
}
if (ABS(norm_in-reference_norm_in) > EPSILON) {
printf("ERROR: L1 input norm         = "FSTR", Reference L1 input norm = "FSTR"\n",
norm_in, reference_norm_in);
validate = 0;
}
else {
#if VERBOSE
printf("SUCCESS: Reference L1 input norm   = "FSTR", L1 input norm   = "FSTR"\n", 
reference_norm_in, norm_in);
#endif
}
full_cycles = ((iterations+1)/(period*4));
leftover_iterations = (iterations+1)%(period*4);
for (g=0; g<4; g++) {
iterations_r[g] = sub_iterations*(full_cycles*duration+
MIN(MAX(0,leftover_iterations-g*period),duration));
reference_norm_r[g] = (DTYPE) (iterations_r[g]) * (COEFX + COEFY);
if (iterations_r[g]==0) {
reference_norm_in_r[g] = 0;
}
else {
bg_updates = (full_cycles*4 + g)*period;
r_updates  = MIN(MAX(0,leftover_iterations-g*period),duration) *
sub_iterations;
if (bg_updates > iterations) {
bg_updates -= 4*period;
r_updates = sub_iterations*duration;
}
reference_norm_in_r[g] = 
(COEFX*G_istart_r[g] + COEFY*G_jstart_r[g]) +
(COEFX+COEFY)*(n_r-1)/2.0 +
(DTYPE) bg_updates +
(DTYPE) r_updates;
}
if (ABS(norm_r[g]-reference_norm_r[g]) > EPSILON) {
printf("ERROR:   Reference L1 norm %d       = "FSTR", L1 norm         = "FSTR"\n",
g, reference_norm_r[g], norm_r[g]);
validate = 0;
}
else {
#if VERBOSE
printf("SUCCESS: Reference L1 norm %d       = "FSTR", L1 norm         = "FSTR"\n", g,
reference_norm_r[g], norm_r[g]);
#endif
}
#if 0
if (ABS(norm_in_r[g]-reference_norm_in_r[g]) > EPSILON) {
printf("ERROR:   Reference L1 input norm %d = "FSTR", L1 input norm %d = "FSTR"\n",
g, g, reference_norm_in_r[g], norm_in_r[g]);
validate = 0;
}
else {
#if VERBOSE
printf("SUCCESS: Reference L1 input norm %d = "FSTR", L1 input norm %d = "FSTR"\n", 
g, reference_norm_in_r[g], g, norm_in_r[g]);
#endif
}
#endif
}
if (!validate) {
printf("Solution does not validate\n");
}
else {
printf("Solution validates\n");
flops = f_active_points_bg * iterations;
iterations_r[0]--;
for (g=0; g<4; g++) flops += f_active_points_r * iterations_r[g];
flops *= (DTYPE) (2*stencil_size+1);
if (refine_level>0) {
num_interpolations--;
flops += n_r_true*(num_interpolations)*3*(n_r_true+n_r);
}
avgtime = stencil_time/iterations;
printf("Rate (MFlops/s): "FSTR"  Avg time (s): %lf\n",
1.0E-06 * flops/stencil_time, avgtime);
}
}
Fenix_Finalize();
MPI_Finalize();
return(MPI_SUCCESS);
}
