#include "hbconvrt.h"
#include "symfac.h"


void ereach_csr(hbmat_t *A, int r, int *etree, vector_t* sub_row){
int m = A->m; int n = A->n; int elemc = A->elemc;
int *vptr = A->vptr; int *vpos = A->vpos; fp_t *vval = A->vval;

vector_clear(sub_row);
vel_t ab_vel;


int k;
for ( k = vptr[r]; k < vptr[r+1]; ++k ) {
ab_vel.i = vpos[k];
vector_insert_t(sub_row, ab_vel);
int inserted = 1;
ab_vel.i = etree[ab_vel.i];
while ( inserted && ab_vel.i > 0 && ab_vel.i <= r) {
inserted = vector_insert_t(sub_row, ab_vel);
ab_vel.i = etree[ab_vel.i];
}
}
vector_qsorti(sub_row);
}

hbmat_t* hb2hbh_sym_etree_csr(hbmat_t *A, int b, int* etree){

int m = A->m; int n = A->n; int elemc = A->elemc;
int *vptr = A->vptr; int *vpos = A->vpos; fp_t* vval = A->vval;
int M = ( m + b - 1 ) / b;
int N = ( n + b - 1 ) / b;
int num = ((1 + M) * N) / 2;

hbmat_t* hyper = malloc(sizeof(hbmat_t));
hyper->m = M; hyper->n = N; hyper->vdiag = NULL;
hyper->vval = (hbmat_t**) malloc(num * sizeof(hbmat_t*));

hbmat_t* acchb = malloc(num*sizeof(hbmat_t));
int acc = 0;

vector_t* ab_vptr = vector_create(); 
vector_t* ab_vpos = vector_create();
vector_clear(ab_vptr); vector_clear(ab_vpos);

vel_t pos_val;

if ( M==0 || N==0 ) {
fprintf( stderr, "block size %i too large\n", b);
}

vector_t **vvptr = malloc(M * sizeof(vector_t*));
vector_t **vvpos = malloc(M * sizeof(vector_t*));
vector_t **vvval = malloc(M * sizeof(vector_t*));
int i;
for ( i = 0; i < M; ++i) {
vvptr[i] = vector_create();
vvpos[i] = vector_create();
vvval[i] = vector_create();
}

vector_t *vpos_tmp = vector_create();
fp_t *vval_tmp = malloc(m * sizeof(fp_t));
int I;
for ( I = 0; I < M; ++I ) {
pos_val.i = ab_vpos->elemc;
vector_insert(ab_vptr, pos_val);

int istart = I*b; int iend = (I+1)*b;  
int blocks = I + 1; 
int J;
for ( J = 0; J < blocks; ++J ) {
vector_clear(vvptr[J]);
vector_clear(vvpos[J]);
vector_clear(vvval[J]);
pos_val.i = 0;
vector_insert(vvptr[J], pos_val);
}
int i;
for ( i = istart; i < iend; ++i ) {  

if ( i >= m ) {
pos_val.i = i - (blocks - 1) * b;
vector_insert(vvpos[blocks-1], pos_val);
pos_val.d = 1;
vector_insert(vvval[blocks-1], pos_val);
int ll;
for ( ll = 0; ll < blocks; ++ll) {
pos_val.i = vvpos[ll]->elemc;
vector_insert(vvptr[ll], pos_val);
}
continue;
}

ereach_csr(A, i, etree, vpos_tmp);
int tmp;
for ( tmp = 0; tmp < vpos_tmp->elemc; ++tmp){
vval_tmp[tmp] = 0;
}
int ptr = 0;

int j;
for ( j = vptr[i]; j < vptr[i+1]; ++j ){
while(vpos_tmp->elem[ptr].i < vpos[j])
++ptr;
if (vpos_tmp->elem[ptr].i == vpos[j]){
vval_tmp[ptr] = vval[j];
++ptr;
}
}
ptr = 0;

int J;
for ( J = 0; J < blocks; ++J ) {
int jstart = J * b; 
int jend = (J + 1) * b;
while ( vpos_tmp->elem[ptr].i < jend && ptr < vpos_tmp->elemc ) {
pos_val.i = vpos_tmp->elem[ptr].i - jstart;
vector_insert(vvpos[J], pos_val);
pos_val.d = vval_tmp[ptr];
vector_insert(vvval[J], pos_val);
++ptr;
}
pos_val.i = vvpos[J]->elemc;
vector_insert(vvptr[J], pos_val);
}
}

for ( J=0; J < blocks; ++J ) {
if ( !vvpos[J]->elemc )
continue;
int vval_ptr;
acchb[acc].m = b; acchb[acc].n = b; acchb[acc].elemc = vvpos[J]->elemc;
acchb[acc].vdiag = NULL;
acchb[acc].vptr = vector2int_nf(vvptr[J]);
acchb[acc].vpos = vector2int_nf(vvpos[J]);
acchb[acc].vval = vector2double_nf(vvval[J]);

vval_ptr = ab_vpos->elemc;
pos_val.i = J;
vector_insert(ab_vpos, pos_val);
((hbmat_t**)hyper->vval)[vval_ptr] = &(acchb[acc]);
++acc;
}
}
pos_val.i = ab_vpos->elemc;
vector_insert(ab_vptr, pos_val);
hyper->elemc = ab_vpos->elemc;
hyper->vptr = vector2int(ab_vptr);
hyper->vpos = vector2int(ab_vpos);

vector_free(vpos_tmp); free(vval_tmp);
for ( i = 0; i < M; ++i) {
vector_free(vvptr[i]);
vector_free(vvpos[i]);
vector_free(vvval[i]);
}
free(vvptr); free(vvpos); free(vvval);

return hyper;
}

void ereach_csr_p(hbmat_t *A, int r, int *etree, vector_t *sub_row, vector_t *sub_val){
int m = A->m; int n = A->n; int elemc = A->elemc;
int *vptr = A->vptr; int *vpos = A->vpos; fp_t *vval = A->vval;

vector_clear(sub_row);
vector_clear(sub_val);
vel_t ab_vel;


int k;
for ( k = vptr[r]; k < vptr[r+1]; ++k ) {
ab_vel.i = vpos[k];
vector_insert_t(sub_row, ab_vel);
int inserted = 1;
ab_vel.i = etree[ab_vel.i];
while ( inserted && ab_vel.i > 0 && ab_vel.i <= r) {
inserted = vector_insert_t(sub_row, ab_vel);
ab_vel.i = etree[ab_vel.i];
}
}
vector_qsorti(sub_row); 

int ccol = vptr[r];
int i;
for ( i = 0; i < sub_row->elemc; ++i ) {
while ( vpos[ccol] < sub_row->elem[i].i && ccol < vptr[r+1] )
++ccol;
if ( vpos[ccol] == sub_row->elem[i].i ){
ab_vel.d = vval[ccol];
vector_insert(sub_val, ab_vel);
continue;
}
if (vpos[ccol] > sub_row->elem[i].i ) {
ab_vel.d = 0;
vector_insert(sub_val, ab_vel);
}
}
}

void symbolic_csr_task(int I, int J, hbmat_t *A, int b, int *etree, int *entry, hbmat_t *block){
int m = A->m;
int* vptr = A->vptr; int* vpos = A->vpos; 
fp_t* vval = A->vval;
int brow = I*b; int erow = (I+1)*b;
int bcol = J*b; int ecol = (J+1)*b;

vector_t* ab_vptr = vector_create();
vector_t* ab_vpos = vector_create();
vector_t* ab_vval = vector_create();
vector_clear(ab_vptr); vector_clear(ab_vpos); vector_clear(ab_vval);
vel_t pos_val;

*entry = 0;

int L;
for ( L = brow; L < erow; ++L ) {
pos_val.i = ab_vpos->elemc; 
vector_insert(ab_vptr, pos_val);

if ( L >= m ) {
if ( ecol < m )
continue;
pos_val.i = L - bcol;
vector_insert(ab_vpos, pos_val);
pos_val.d = 1;
vector_insert(ab_vval, pos_val);
continue;
}

int p_elemc = ab_vpos->elemc;
int k;
for ( k = vptr[L]; k < vptr[L+1]; ++k ) {

if ( vpos[k] >= bcol && vpos[k] < ecol && vpos[k] <= L){
pos_val.i = vpos[k] - bcol;
vector_insert_t_partial(ab_vpos, pos_val, p_elemc);
}
int inserted = 1;
pos_val.i = etree[vpos[k]];
while ( inserted && pos_val.i < ecol && pos_val.i <= L) {
if ( pos_val.i >= bcol ) {
vel_t tmp;
tmp.i = pos_val.i - bcol;
inserted = vector_insert_t_partial(ab_vpos, tmp, p_elemc);
}
pos_val.i = etree[pos_val.i];
}
}

vector_qsorti_partial(ab_vpos, p_elemc);

int ccol = vptr[L];
int i;
for ( i = p_elemc; i < ab_vpos->elemc; ++i ) {
while ( (vpos[ccol] - bcol) < ab_vpos->elem[i].i && ccol < vptr[L+1])
++ccol;

if ( (vpos[ccol] - bcol) == ab_vpos->elem[i].i ){
pos_val.d = vval[ccol];
vector_insert(ab_vval, pos_val);
continue;
}
else {
pos_val.d = 0;
vector_insert(ab_vval, pos_val);
}
}
}

pos_val.i = ab_vpos->elemc;
vector_insert(ab_vptr, pos_val);

if ( ab_vpos->elemc ){
block->m = b; block->n = b; block->elemc = ab_vpos->elemc;
block->vdiag = NULL;
block->vptr = vector2int(ab_vptr); 
block->vpos = vector2int(ab_vpos);
block->vval = vector2double(ab_vval);
*entry = 1;
}
}

hbmat_t* hb2hbh_sym_etree_csr_p(hbmat_t *A, int b, int *etree){

int m = A->m; int n = A->n; int elemc = A->elemc;
int *vptr = A->vptr; int *vpos = A->vpos; fp_t* vval = A->vval;
int M = ( m + b - 1 ) / b;
int N = ( n + b - 1 ) / b;
int num = ((1 + M) * N) / 2;

hbmat_t* hyper = malloc(sizeof(hbmat_t));
hyper->m = M; hyper->n = N; hyper->vdiag = NULL;
hyper->vval = malloc(num * sizeof(hbmat_t*));
hbmat_t** hbmat_array = malloc(num * sizeof(hbmat_t*));
int* hentry = malloc(num * sizeof(int));

vector_t* ab_vptr = vector_create(); 
vector_t* ab_vpos = vector_create();
vector_clear(ab_vptr); vector_clear(ab_vpos);
vel_t pos_val;

if ( M==0 || N==0 ) {
fprintf( stderr, "block size %i too large\n", b);
}

int i;
for ( i = 0; i < num; ++i ) {
hbmat_array[i] = malloc(sizeof(hbmat_t));
}
int acc = 0;
int I, J;
for ( I = 0; I < M; ++I ){
for ( J = 0; J < I+1; ++J){
symbolic_csr_task(I, J, A, b, etree, &(hentry[acc]), hbmat_array[acc]);
++acc;
}
}

#pragma omp taskwait

acc = 0;
int acc0 = 0;
for ( I = 0; I < M; ++I ) {
pos_val.i = ab_vpos->elemc;
vector_insert(ab_vptr, pos_val);
for ( J = 0; J < I+1; ++J ) {
if ( hentry[acc] ) {
pos_val.i = J;
vector_insert(ab_vpos, pos_val);
((hbmat_t**)hyper->vval)[acc0] = hbmat_array[acc];
++acc;
++acc0;
} else {
free(hbmat_array[acc]);
++acc;
}
}
}

pos_val.i = ab_vpos->elemc;
vector_insert(ab_vptr, pos_val);
hyper->elemc = ab_vpos->elemc;
hyper->vptr = vector2int(ab_vptr);
hyper->vpos = vector2int(ab_vpos);

return hyper;
}

void hyper_sym_csr_task0(int I, int J, hbmat_t *A, int b, int *etree, int *entry){
int m = A->m;
int* vptr = A->vptr; int* vpos = A->vpos; 
int brow = I*b; int erow = (I+1)*b;
int bcol = J*b; int ecol = (J+1)*b;
int ccol;

*entry = 0;


if (erow >= m)
erow = m;

int L;
for ( L = brow; L < erow; ++L ) {
int k;
for ( k = vptr[L]; k < vptr[L+1]; ++k ) {
ccol = vpos[k];
while ( ccol < ecol && ccol <= L ) {
if ( ccol >= bcol && ccol < ecol){
*entry = 1;
return;
}
ccol = etree[ccol];
}
}
}
}

void hyper_sym_csr_task1(hbmat_t *block){


int b = block->m;
int I = block->orig_row; int J = block->orig_col;
hbmat_t* A = block->orig;
int* etree = block->e_tree;


int m = A->m;
int* vptr = A->vptr; int* vpos = A->vpos; 
fp_t* vval = A->vval;
int brow = I*b; int erow = (I+1)*b;
int bcol = J*b; int ecol = (J+1)*b;

vector_t* ab_vptr = vector_create();
vector_t* ab_vpos = vector_create();
vector_t* ab_vval = vector_create();
vector_clear(ab_vptr); vector_clear(ab_vpos); vector_clear(ab_vval);
vel_t pos_val;

int L;
for ( L = brow; L < erow; ++L ) {
pos_val.i = ab_vpos->elemc; 
vector_insert(ab_vptr, pos_val);


if ( L >= m ) {
if ( ecol < m )
continue;
pos_val.i = L - bcol;
vector_insert(ab_vpos, pos_val);
pos_val.d = 1;
vector_insert(ab_vval, pos_val);
continue;
}

int p_elemc = ab_vpos->elemc;
int k;
for ( k=vptr[L]; k < vptr[L+1]; ++k ) {

if ( vpos[k] >= bcol && vpos[k] < ecol && vpos[k] <= L){
pos_val.i = vpos[k] - bcol;
vector_insert_t_partial(ab_vpos, pos_val, p_elemc);
}
int inserted = 1;
pos_val.i = etree[vpos[k]];
while ( inserted && pos_val.i < ecol && pos_val.i <= L) {
if ( pos_val.i >= bcol ) {
vel_t tmp;
tmp.i = pos_val.i - bcol;
inserted = vector_insert_t_partial(ab_vpos, tmp, p_elemc);
}
pos_val.i = etree[pos_val.i];
}
}

vector_qsorti_partial(ab_vpos, p_elemc);

int ccol = vptr[L];
int i;
for ( i=p_elemc; i < ab_vpos->elemc; ++i ) {
while ( (vpos[ccol] - bcol) < ab_vpos->elem[i].i && ccol < vptr[L+1])
++ccol;

if ( (vpos[ccol] - bcol) == ab_vpos->elem[i].i ){
pos_val.d = vval[ccol];
vector_insert(ab_vval, pos_val);
continue;
}
else {		
pos_val.d = 0;
vector_insert(ab_vval, pos_val);
}
}
}

pos_val.i = ab_vpos->elemc;
vector_insert(ab_vptr, pos_val);

if ( ab_vpos->elemc ){
block->m = b; block->n = b; block->elemc = ab_vpos->elemc;
block->vdiag = NULL;
block->vptr = vector2int(ab_vptr); 
block->vpos = vector2int(ab_vpos);
block->vval = vector2double(ab_vval);
}else
printf("Warning! task1 fail. I %d J %d\n", I, J);
}

void hyper_sym_csr_task2(hbmat_t *B) {

int b = B->m;
int I = B->orig_row; int J = B->orig_col;
hbmat_t *A = B->orig;
int *etree = B->e_tree;

int m = A->m;
int *vptr = A->vptr; int* vpos = A->vpos; 
fp_t *vval = A->vval;
int brow = I*b; int erow = (I+1)*b;
int bcol = J*b; int ecol = (J+1)*b;

hbmat_t *H = B->hyper;
pthread_mutex_lock(H->mtx);
B->vptr = H->vptr_pool + H->vptr_pp * H->vptr_unit;
B->vpos = H->vpos_pool + H->vpos_pp * H->vpos_unit;
B->vval = H->vval_pool + H->vval_pp * H->vval_unit;
H->vptr_pp++; H->vpos_pp++; H->vval_pp++;
pthread_mutex_unlock(H->mtx);

vector_int* ab_vptr = vector_int_create(B->vptr, H->vptr_unit);
vector_int* ab_vpos = vector_int_create(B->vpos, H->vpos_unit);
vector_double* ab_vval = vector_double_create(B->vval, H->vval_unit);
int val_int; fp_t val_fp;


int *nz_map = calloc(b, sizeof(int));

int L;
for ( L = brow; L < erow; ++L ) {
int Lp1 = L + 1;
val_int = ab_vpos->elemc; 
vector_int_insert(ab_vptr, val_int);

int p_elemc = ab_vpos->elemc;
int k = vptr[L];

while ( vpos[k] < ecol && k < vptr[Lp1] ) {
if ( vpos[k] >= bcol && vpos[k] <= L){
val_int = vpos[k] - bcol;
if ( nz_map[val_int] < Lp1 ) {
vector_int_insert(ab_vpos, val_int);
nz_map[val_int] = Lp1;
}
}
int merged = 0;
val_int = etree[vpos[k]];
while ( !merged && val_int < ecol && val_int <= L) {
if ( val_int >= bcol ) {
int tmp;
tmp = val_int - bcol;
if ( nz_map[tmp] < Lp1 ){
vector_int_insert(ab_vpos, tmp);
nz_map[tmp] = Lp1;
}else
merged = 1;
}
val_int = etree[val_int];
}
++k;
}
vector_int_qsorti_partial(ab_vpos, p_elemc);

int ccol = vptr[L];
int cend = vptr[Lp1];
int i;
for ( i = p_elemc; i < ab_vpos->elemc; ++i ) {
int lj = vpos[ccol] - bcol;
while ( lj < ab_vpos->elem[i] && ccol < cend ) {
lj = vpos[++ccol] - bcol;
}

fp_t val_fp = 0.0;
int nonzero = lj == ab_vpos->elem[i] && ccol < cend;
if ( nonzero ) {
val_fp = vval[ccol];
}

vector_double_insert(ab_vval, val_fp);
}
}
free(nz_map);

val_int = ab_vpos->elemc;
vector_int_insert(ab_vptr, val_int);
B->elemc = ab_vpos->elemc;
}

hbmat_t* hb2hbh_sym_etree(hbmat_t *A, int b, int* etree){
int m = A->m; int n = A->n; int elemc = A->elemc;
int *vptr = A->vptr; int *vpos = A->vpos; fp_t* vval = A->vval;
int M = ( m + b - 1 ) / b;
int N = ( n + b - 1 ) / b;

hbmat_t* Ab = malloc(sizeof(hbmat_t));
Ab->m = M; Ab->n = N; Ab->elemc = 0;
Ab->vdiag = NULL;


int num = ((1 + M) * N) / 2; 
hbmat_t* acchb = malloc(num * sizeof(hbmat_t));
int acc = 0 ;
Ab->vval = malloc(num * sizeof(hbmat_t*));
int ab_count = 0 ;
vector_t *ab_vptr, *ab_vpos;
ab_vptr = vector_create(); ab_vpos = vector_create();
vector_clear(ab_vptr); vector_clear(ab_vpos);

vector_t** vec_col = (vector_t**) malloc(n*sizeof(vector_t*));
vel_t ab_vel;

if ( M==0 || N==0 ) {
fprintf( stderr, "block size %i too large\n", b);
}

int i;
for ( i = 0; i < m; i++) {
vec_col[i] = vector_create();
vector_clear(vec_col[i]);
}

int J;
for ( J = 0; J < N; J++) {
int jstart = J * b;
int jc = n - jstart;
jc = jc < b ? jc: b; 
ab_vel.i = ab_vpos->elemc;
vector_insert(ab_vptr, ab_vel);

int j;
for ( j = jstart; j < jstart+jc; j++) {
int k;
for ( k = vptr[j]; k < vptr[j+1]; k++) {
ab_vel.i = vpos[k];
vector_insert_t(vec_col[j], ab_vel);
}
vector_qsorti(vec_col[j]);
int l;
for ( l = 2; l < vec_col[j]->elemc; l++) {
ab_vel.i = vec_col[j]->elem[l].i;
vector_insert_t(vec_col[vec_col[j]->elem[1].i], ab_vel);
}
}

int I;
for ( I = J; I < M; I++ ) {
int base_col = J * b ;
int base_row = I * b ;
int ic = m - base_row;
int max_row = base_row+b;

ic = ic < b ? ic : b;
vector_t *sub_col, *sub_vptr, *sub_vpos, *sub_vval;
vel_t acchb_vel;
sub_col = vector_create_size(jc);
sub_vptr = vector_create_size(jc);
sub_vpos = vector_create_size(jc);
sub_vval = vector_create_size(jc);
vector_clear(sub_vptr);
vector_clear(sub_vpos);
vector_clear(sub_vval);

int j;
for ( j = 0; j < b; j++ ) {
int current_col = base_col+j; 
vector_clear(sub_col);
ab_vel.i = sub_vpos->elemc;
vector_insert(sub_vptr, ab_vel);

if (current_col >= n){
acchb_vel.i = current_col-base_row;
vector_insert(sub_vpos, acchb_vel);
acchb_vel.d = 1;
vector_insert(sub_vval, acchb_vel);
continue;
}

int k;
for ( k = 0; k < vec_col[current_col]->elemc; k++) {
if (vec_col[current_col]->elem[k].i < base_row)
continue;
if (vec_col[current_col]->elem[k].i >= max_row)
break;
acchb_vel.i = vec_col[current_col]->elem[k].i;
vector_insert(sub_col, acchb_vel);
}

for ( k = 0; k < sub_col->elemc; k++ ) {
acchb_vel.i = sub_col->elem[k].i-base_row;
vector_insert(sub_vpos, acchb_vel);
int l;
for ( l = vptr[current_col]; l < vptr[current_col+1]; l++) {
acchb_vel.d = 0;
if(sub_col->elem[k].i == vpos[l]){
acchb_vel.d = vval[l];
break;
}
}
vector_insert(sub_vval, acchb_vel);
}
}
if (sub_vpos->elemc != 0){
ab_vel.i = sub_vpos->elemc;
vector_insert(sub_vptr, ab_vel);
acchb[ab_vpos->elemc].m = b;
acchb[ab_vpos->elemc].n = b;
acchb[ab_vpos->elemc].elemc = sub_vpos->elemc;
acchb[ab_vpos->elemc].vptr = vector2int(sub_vptr);
acchb[ab_vpos->elemc].vpos = vector2int(sub_vpos);
acchb[ab_vpos->elemc].vval = vector2double(sub_vval);
acchb[ab_vpos->elemc].vdiag = NULL;
((hbmat_t**)Ab->vval)[ab_vpos->elemc] = acchb + ab_vpos->elemc;
ab_vel.i = I;
vector_insert(ab_vpos, ab_vel);
}
else{
vector_free(sub_vptr);
vector_free(sub_vpos);
vector_free(sub_vval);
}
vector_free(sub_col);
}
}

for ( i = 0; i < n; i++) {
vector_free(vec_col[i]);
}
free(vec_col);

ab_vel.i = ab_vpos->elemc;
vector_insert(ab_vptr, ab_vel);
Ab->elemc = ab_vpos->elemc;
Ab->vptr = vector2int(ab_vptr);
Ab->vpos = vector2int(ab_vpos);

return Ab;

}

hbmat_t *hbh2hb_sym (hbmat_t *A){
hbmat_t *B = (hbmat_t*) malloc(sizeof(hbmat_t));
int M = A->m; int N = A->n;
int elemc = A->elemc;
int* vptr = A->vptr;
int* vpos = A->vpos;
hbmat_t** vval = A->vval;

vector_t *b_vptr, *b_vpos, *b_vval;
b_vptr = vector_create(); b_vpos = vector_create(); b_vval = vector_create();
vector_clear(b_vptr); vector_clear(b_vpos); vector_clear(b_vval);
vel_t b_vptr_vel, b_vpos_vel, b_vval_vel;
hbmat_t* sub_matrix;
int bs = vval[0]->m; 
int col_counter = 0;

int J;
for ( J = 0; J < N; J++ ) {
sub_matrix = vval[vptr[J]]; 
int tot_col = sub_matrix->n;
int j;
for ( j = 0; j < tot_col; j++ ) {
col_counter++;
b_vptr_vel.i = b_vpos->elemc;
vector_insert(b_vptr, b_vptr_vel);
int I;
for ( I = vptr[J]; I < vptr[J+1]; I++ ) {
int c_row = vpos[I];
int row_offset = c_row*bs;
sub_matrix = vval[I];
int jj;
for( jj = sub_matrix->vptr[j]; jj < sub_matrix->vptr[j+1]; jj++ ) {
if(1 || ((fp_t*)sub_matrix->vval)[jj] != 0 ){
b_vpos_vel.i = sub_matrix->vpos[jj] + row_offset;
vector_insert(b_vpos, b_vpos_vel);
b_vval_vel.d = ((fp_t*)sub_matrix->vval)[jj];
vector_insert(b_vval, b_vval_vel);
}
}
}
}
}

b_vptr_vel.i = b_vpos->elemc;
vector_insert(b_vptr, b_vptr_vel);


B->m = B->n = col_counter;
B->elemc = b_vpos->elemc;
B->vptr = vector2int(b_vptr);
B->vpos = vector2int(b_vpos);
B->vval = vector2double(b_vval);
return B;
}
