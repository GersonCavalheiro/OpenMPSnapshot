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
void get_BG_data(int load_balance, DTYPE *in_bg, DTYPE *ing_r, int my_ID, long expand,
int Num_procs, long L_width_bg, 
long L_istart_bg, long L_iend_bg, long L_jstart_bg, long L_jend_bg,
long L_istart_r, long L_iend_r, long L_jstart_r, long L_jend_r,
long G_istart_r, long G_jstart_r, MPI_Comm comm_bg, MPI_Comm comm_r,
long L_istart_r_gross, long L_iend_r_gross, 
long L_jstart_r_gross, long L_jend_r_gross, 
long L_width_r_true_gross, long L_istart_r_true_gross, long L_iend_r_true_gross,
long L_jstart_r_true_gross, long L_jend_r_true_gross, int g) {
long send_vec[8], *recv_vec, offset, i, j, p, acc_send, acc_recv;
int *recv_offset, *recv_count, *send_offset, *send_count;
DTYPE *recv_buf, *send_buf;
if (load_balance == no_talk) {
if (comm_r != MPI_COMM_NULL) {
for (j=L_jstart_r_gross; j<=L_jend_r_gross; j++) 
for (i=L_istart_r_gross; i<=L_iend_r_gross; i++) {
int ir = i-G_istart_r, jr = j-G_jstart_r;
ING_R(ir*expand,jr*expand) = IN(i,j);
}
}
}
else {
recv_vec    = (long *)  prk_malloc(sizeof(long)*Num_procs*8);
recv_count  = (int *)   prk_malloc(sizeof(int)*Num_procs);
recv_offset = (int *)   prk_malloc(sizeof(int)*Num_procs);
send_count  = (int *)   prk_malloc(sizeof(int)*Num_procs);
send_offset = (int *)   prk_malloc(sizeof(int)*Num_procs);
if (!recv_vec || !recv_count || !recv_offset || !send_count || !send_offset){
printf("ERROR: Could not allocate space for Allgather on rank %d\n", my_ID);
MPI_Abort(MPI_COMM_WORLD, 66); 
}
send_vec[0] = L_istart_bg;
send_vec[1] = L_iend_bg;
send_vec[2] = L_jstart_bg;
send_vec[3] = L_jend_bg;
send_vec[4] = L_istart_r_gross;
send_vec[5] = L_iend_r_gross;
send_vec[6] = L_jstart_r_gross;
send_vec[7] = L_jend_r_gross;
MPI_Allgather(send_vec, 8, MPI_LONG, recv_vec, 8, MPI_LONG, MPI_COMM_WORLD);
acc_recv = 0;
for (acc_recv=0,p=0; p<Num_procs; p++) {
recv_vec[p*8+0] = MAX(recv_vec[p*8+0], L_istart_r_gross); 
recv_vec[p*8+1] = MIN(recv_vec[p*8+1], L_iend_r_gross);
recv_vec[p*8+2] = MAX(recv_vec[p*8+2], L_jstart_r_gross);
recv_vec[p*8+3] = MIN(recv_vec[p*8+3], L_jend_r_gross);
recv_count[p] = MAX(0,(recv_vec[p*8+1]-recv_vec[p*8+0]+1)) *
MAX(0,(recv_vec[p*8+3]-recv_vec[p*8+2]+1));
acc_recv += recv_count[p];
}
if (acc_recv) {
recv_buf = (DTYPE *) prk_malloc(sizeof(DTYPE)*acc_recv);
if (!recv_buf) {
printf("ERROR: Could not allocate space for recv_buf on rank %d\n", my_ID);
MPI_Abort(MPI_COMM_WORLD, 66); 
}
}
for (acc_send=0,p=0; p<Num_procs; p++) {
recv_vec[p*8+4] = MAX(recv_vec[p*8+4], L_istart_bg);
recv_vec[p*8+5] = MIN(recv_vec[p*8+5], L_iend_bg);
recv_vec[p*8+6] = MAX(recv_vec[p*8+6], L_jstart_bg);
recv_vec[p*8+7] = MIN(recv_vec[p*8+7], L_jend_bg);
send_count[p] = MAX(0,(recv_vec[p*8+5]-recv_vec[p*8+4]+1)) *
MAX(0,(recv_vec[p*8+7]-recv_vec[p*8+6]+1));
acc_send += send_count[p]; 
}
if (acc_send) {
send_buf    = (DTYPE *) prk_malloc(sizeof(DTYPE)*acc_send);
if (!send_buf) {
printf("ERROR: Could not allocate space for send_buf on rank %d\n", my_ID);
MPI_Abort(MPI_COMM_WORLD, 66); 
}
}
recv_offset[0] =  send_offset[0] = 0;
for (p=1; p<Num_procs; p++) {
recv_offset[p] = recv_offset[p-1]+recv_count[p-1];
send_offset[p] = send_offset[p-1]+send_count[p-1];
}
offset = 0;
if (comm_bg != MPI_COMM_NULL) for (p=0; p<Num_procs; p++){
if (recv_vec[p*8+4]<=recv_vec[p*8+5]) { 
for (j=recv_vec[p*8+6]; j<=recv_vec[p*8+7]; j++) {
for (i=recv_vec[p*8+4]; i<=recv_vec[p*8+5]; i++){
send_buf[offset++] = IN(i,j);
}
}
}
}
MPI_Alltoallv(send_buf, send_count, send_offset, MPI_DTYPE, 
recv_buf, recv_count, recv_offset, MPI_DTYPE, MPI_COMM_WORLD);
offset = 0;
if (comm_r != MPI_COMM_NULL) for (p=0; p<Num_procs; p++) {
if (recv_vec[p*8+0]<=recv_vec[p*8+1]) { 
for (j=recv_vec[p*8+2]-G_jstart_r; j<=recv_vec[p*8+3]-G_jstart_r; j++) {
for (i=recv_vec[p*8+0]-G_istart_r; i<=recv_vec[p*8+1]-G_istart_r; i++) {
ING_R(i*expand,j*expand) = recv_buf[offset++];
}
}
}
}
prk_free(recv_vec);
prk_free(recv_count);
prk_free(recv_offset);
prk_free(send_count);
prk_free(send_offset);
if (acc_recv) prk_free(recv_buf);
if (acc_send) prk_free(send_buf);
}
}
