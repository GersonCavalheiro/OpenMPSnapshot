#include<inttypes.h>
#include<math.h>
#include<string.h>
#include<limits.h>
#include<stdarg.h>
#include<ctype.h>
#include "macros.h"
#include "utils.h"
#ifdef __MACH__ 
#include <mach/mach_time.h> 
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
void get_max_float(const int64_t ND1, const float *cz1, float *czmax)
{
float max=*czmax;
for(int64_t i=0;i<ND1;i++) {
if(cz1[i] > max) max = cz1[i];
}
*czmax = max;
}
void get_max_double(const int64_t ND1, const double *cz1, double *czmax)
{
double max=*czmax;
for(int64_t i=0;i<ND1;i++) {
if(cz1[i] > max) max = cz1[i];
}
*czmax = max;
}
int setup_bins(const char *fname,double *rmin,double *rmax,int *nbin,double **rupp)
{
const int MAXBUFSIZE=1000;
char buf[MAXBUFSIZE];
FILE *fp=NULL;
double low,hi;
const char comment='#';
const int nitems=2;
int nread=0;
*nbin = ((int) getnumlines(fname,comment))+1;
*rupp = my_calloc(sizeof(double),*nbin+1);
fp = my_fopen(fname,"r");
if(fp == NULL) {
free(*rupp);
return EXIT_FAILURE;
}
int index=1;
while(1) {
if(fgets(buf,MAXBUFSIZE,fp)!=NULL) {
nread=sscanf(buf,"%lf %lf",&low,&hi);
if(nread==nitems) {
if(index==1) {
*rmin=low;
(*rupp)[0]=low;
}
(*rupp)[index] = hi;
index++;
}
} else {
break;
}
}
*rmax = (*rupp)[index-1];
fclose(fp);
(*rupp)[*nbin]=*rmax ;
(*rupp)[*nbin-1]=*rmax ;
return EXIT_SUCCESS;
}
int setup_bins_double(const char *fname,double *rmin,double *rmax,int *nbin,double **rupp)
{
const int MAXBUFSIZE=1000;
char buf[MAXBUFSIZE];
double low,hi;
const char comment='#';
const int nitems=2;
int nread=0;
*nbin = ((int) getnumlines(fname,comment))+1;
*rupp = my_calloc(sizeof(double),*nbin+1);
FILE *fp = my_fopen(fname,"r");
if(fp == NULL) {
free(*rupp);
return EXIT_FAILURE;
}
int index=1;
while(1) {
if(fgets(buf,MAXBUFSIZE,fp)!=NULL) {
nread=sscanf(buf,"%lf %lf",&low,&hi);
if(nread==nitems) {
if(index==1) {
*rmin=low;
(*rupp)[0]=low;
}
(*rupp)[index] = hi;
index++;
}
} else {
break;
}
}
*rmax = (*rupp)[index-1];
fclose(fp);
(*rupp)[*nbin]=*rmax ;
(*rupp)[*nbin-1]=*rmax ;
return EXIT_SUCCESS;
}
int setup_bins_float(const char *fname,float *rmin,float *rmax,int *nbin,float **rupp)
{
const int MAXBUFSIZE=1000;
char buf[MAXBUFSIZE];
float low,hi;
const char comment='#';
const int nitems=2;
int nread=0;
*nbin = ((int) getnumlines(fname,comment))+1;
*rupp = my_calloc(sizeof(float),*nbin+1);
FILE *fp = my_fopen(fname,"r");
if(fp == NULL) {
free(*rupp);
return EXIT_FAILURE;
}
int index=1;
while(1) {
if(fgets(buf,MAXBUFSIZE,fp)!=NULL) {
nread=sscanf(buf,"%f %f",&low,&hi);
if(nread==nitems) {
if(index==1) {
*rmin=low;
(*rupp)[0]=low;
}
(*rupp)[index] = hi;
index++;
}
} else {
break;
}
}
*rmax = (*rupp)[index-1];
fclose(fp);
(*rupp)[*nbin]=*rmax ;
(*rupp)[*nbin-1]=*rmax ;
return EXIT_SUCCESS;
}
int run_system_call(const char *execstring)
{
int status=system(execstring);
if(status != EXIT_SUCCESS) {
fprintf(stderr,"ERROR: executing system command: \n`%s'\n\n",execstring);
perror(NULL);
}
return EXIT_FAILURE;
}
FILE * my_fopen(const char *fname,const char *mode)
{
FILE *fp = fopen(fname,mode);
if(fp == NULL){
fprintf(stderr,"Could not open file `%s'\n",fname);
perror(NULL);
}
return fp;
}
FILE * my_fopen_carefully(const char *fname,void (*header)(FILE *))
{
FILE *fp = fopen(fname,"r");
if(fp == NULL) {
fp = my_fopen(fname,"w");
if(fp != NULL) {
(*header)(fp);
}
} else {
fclose(fp);
fp = my_fopen(fname,"a+");
}
return fp;
}
size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
size_t nwritten;
nwritten = fwrite(ptr, size, nmemb, stream);
if(nwritten != nmemb){
fprintf(stderr,"I/O error (fwrite) has occured.\n");
fprintf(stderr,"Instead of reading nmemb=%zu, I got nread = %zu \n",nmemb,nwritten);
perror(NULL);
return -1;
}
return nwritten;
}
size_t my_fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
size_t nread;
nread = fread(ptr, size, nmemb, stream);
if(nread != nmemb) {
fprintf(stderr,"I/O error (fread) has occured.\n");
fprintf(stderr,"Instead of reading nmemb=%zu, I got nread = %zu\n",nmemb,nread);
perror(NULL);
return -1;
}
return nread;
}
int my_fseek(FILE *stream, long offset, int whence)
{
int err=fseek(stream,offset,whence);
if(err != 0) {
fprintf(stderr,"ERROR: Could not seek `%ld' bytes into the file..exiting\n",offset);
perror(NULL);
}
return err;
}
int my_snprintf(char *buffer,int len,const char *format, ...)
{
va_list args;
int nwritten=0;
va_start(args,format);
nwritten=vsnprintf(buffer, (size_t) len, format, args );
va_end(args);
if (nwritten > len || nwritten < 0) {
fprintf(stderr,"ERROR: printing to string failed (wrote %d characters while only %d characters were allocated)\n",nwritten,len);
fprintf(stderr,"Increase `len'=%d in the header file\n",len);
return -1;
}
return nwritten;
}
int is_big_endian(void)
{
union {
uint32_t i;
char c[4];
} e = { 0x01000000 };
return e.c[0];
}
void byte_swap(char * const in, const size_t size, char *out)
{
if(size > 16) {
fprintf(stderr,"WARNING: In %s> About to byte_swap %zu bytes but no intrinsic C data-type exists with size larger than 16 bytes",
__FUNCTION__, size);
}
char *in_char = (char *) in + (size - 1UL);
char *out_char = out;
for(size_t i=0;i<size;i++) {
*out_char = *in_char;
out_char++;
in_char--;
}
}
char * int2bin(int a, char *buffer, int buf_size)
{
buffer += (buf_size - 1);
for (int i = 31; i >= 0; i--) {
*buffer-- = (a & 1) + '0';
a >>= 1;
}
return buffer;
}
void current_utc_time(struct timespec *ts)
{
#ifdef __MACH__ 
static mach_timebase_info_data_t    sTimebaseInfo = {.numer=0, .denom=0};
uint64_t start = mach_absolute_time();
if ( sTimebaseInfo.denom == 0 ) {
mach_timebase_info(&sTimebaseInfo);
}
ts->tv_sec = 0;
ts->tv_nsec = start * sTimebaseInfo.numer / sTimebaseInfo.denom;
#if 0
clock_serv_t cclock;
mach_timespec_t mts;
host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
clock_get_time(cclock, &mts);
mach_port_deallocate(mach_task_self(), cclock);
ts->tv_sec = mts.tv_sec;
ts->tv_nsec = mts.tv_nsec;
#endif
#else
clock_gettime(CLOCK_REALTIME, ts);
#endif
}
char * get_time_string(struct timeval t0,struct timeval t1)
{
const size_t MAXLINESIZE = 1024;
char *time_string = my_malloc(sizeof(char), MAXLINESIZE);
double timediff = t1.tv_sec - t0.tv_sec;
double ratios[] = {24*3600.0,  3600.0,  60.0,  1};
if(timediff < ratios[2]) {
my_snprintf(time_string, MAXLINESIZE,"%6.3lf secs",1e-6*(t1.tv_usec-t0.tv_usec) + timediff);
}  else {
double timeleft = timediff;
size_t curr_index = 0;
int which = 0;
while (which < 4) {
double time_to_print = floor(timeleft/ratios[which]);
if (time_to_print > 1) {
timeleft -= (time_to_print*ratios[which]);
char units[4][10]  = {"days", "hrs" , "mins", "secs"};
char tmp[MAXLINESIZE];
my_snprintf(tmp, MAXLINESIZE, "%5d %s",(int)time_to_print,units[which]);
const size_t len = strlen(tmp);
const size_t required_len = curr_index + len + 1;
XRETURN(MAXLINESIZE >= required_len, NULL,
"buffer overflow will occur: string has space for %zu bytes while concatenating requires at least %zu bytes\n",
MAXLINESIZE, required_len);
strcpy(time_string + curr_index, tmp);
curr_index += len;
}
which++;
}
}
return time_string;
}
void print_time(struct timeval t0,struct timeval t1,const char *s)
{
double timediff = t1.tv_sec - t0.tv_sec;
double ratios[] = {24*3600.0,  3600.0,  60.0,  1};
fprintf(stderr,"Time taken to execute '%s'  = ",s);
if(timediff < ratios[2]) {
fprintf(stderr,"%6.3lf secs",1e-6*(t1.tv_usec-t0.tv_usec) + timediff);
}  else {
double timeleft = timediff;
int which = 0;
while (which < 4) {
double time_to_print = floor(timeleft/ratios[which]);
if (time_to_print > 1) {
char units[4][10]  = {"days", "hrs" , "mins", "secs"};
timeleft -= (time_to_print*ratios[which]);
fprintf(stderr,"%5d %s",(int)time_to_print,units[which]);
}
which++;
}
}
fprintf(stderr,"\n");
}
void* my_realloc(void *x,size_t size,int64_t N,const char *varname)
{
void *tmp = realloc(x,N*size);
if (tmp==NULL) {
fprintf(stderr,"ERROR: Could not reallocate for %"PRId64" elements with %zu size for variable `%s' ..aborting\n",N,size,varname);
perror(NULL);
}
return tmp;
}
void* my_malloc(size_t size,int64_t N)
{
void *x = malloc(N*size);
if (x==NULL){
fprintf(stderr,"malloc for %"PRId64" elements with %zu bytes failed...\n",N,size);
perror(NULL);
}
return x;
}
void* my_calloc(size_t size,int64_t N)
{
void *x = calloc((size_t) N, size);
if (x==NULL)    {
fprintf(stderr,"malloc for %"PRId64" elements with %zu size failed...\n",N,size);
perror(NULL);
}
return x;
}
void my_free(void ** x)
{
if(*x!=NULL)
free(*x);
*x=NULL;
}
void **matrix_malloc(size_t size,int64_t nrow,int64_t ncol)
{
void **m = (void **) my_malloc(sizeof(void *),nrow);
if(m == NULL) {
return NULL;
}
for(int i=0;i<nrow;i++) {
m[i] = (void *) my_malloc(size,ncol);
if(m[i] == NULL) {
for(int j=i-1;j>=0;j--) {
free(m[j]);
}
free(m);
return NULL;
}
}
return m;
}
void **matrix_calloc(size_t size,int64_t nrow,int64_t ncol)
{
void **m = (void **) my_calloc(sizeof(void *),nrow);
if(m == NULL) {
return m;
}
for(int i=0;i<nrow;i++) {
m[i] = (void *) my_calloc(size,ncol);
if(m[i] == NULL) {
for(int j=i-1;j>=0;j--) {
free(m[j]);
}
free(m);
return NULL;
}
}
return m;
}
int matrix_realloc(void **matrix, size_t size, int64_t nrow, int64_t ncol){
void *tmp;
for(int i = 0; i < nrow; i++){
tmp = my_realloc(matrix[i], size, ncol, "matrix_realloc");
if(tmp == NULL){
return EXIT_FAILURE;
}
matrix[i] = tmp;
}
return EXIT_SUCCESS;
}
void matrix_free(void **m,int64_t nrow)
{
if(m == NULL)
return;
for(int i=0;i<nrow;i++)
free(m[i]);
free(m);
}
void *** volume_malloc(size_t size,int64_t nrow,int64_t ncol,int64_t nframe)
{
void ***v = (void ***) my_malloc(sizeof(void **),nrow);
if( v == NULL) {
return NULL;
}
for(int i=0;i<nrow;i++) {
v[i] = (void *) my_malloc(sizeof(void *),ncol);
if(v[i] == NULL) {
for(int jj=i-1;jj>=0;jj--) {
for(int k=0;k<ncol;k++) {
free(v[jj][k]);
}
}
free(v);
return NULL;
}
for(int j=0;j<ncol;j++) {
v[i][j] = my_malloc(size,nframe);
if(v[i][j] == NULL) {
for(int k=ncol-1;k>=0;k--) {
free(v[i][k]);
}
for(int jj=i-1;jj>=0;jj--) {
for(int k=0;k<ncol;k++) {
free(v[jj][k]);
}
}
free(v);
return NULL;
}
}
}
return v;
}
void *** volume_calloc(size_t size,int64_t nrow,int64_t ncol,int64_t nframe)
{
void ***v = (void ***) my_malloc(sizeof(void **),nrow);
if(v == NULL) {
return NULL;
}
for(int i=0;i<nrow;i++) {
v[i] = (void *) my_malloc(sizeof(void *),ncol);
if(v[i] == NULL) {
for(int jj=i-1;jj>=0;jj--) {
for(int k=0;k<ncol;k++) {
free(v[jj][k]);
}
}
free(v);
return NULL;
}
for(int j=0;j<ncol;j++) {
v[i][j] = my_calloc(size,nframe);
if(v[i][j] == NULL) {
for(int k=ncol-1;k>=0;k--) {
free(v[i][k]);
}
for(int jj=i-1;jj>=0;jj--) {
for(int k=0;k<ncol;k++) {
free(v[j][k]);
}
}
free(v);
return NULL;
}
}
}
return v;
}
void volume_free(void ***v,int64_t nrow,int64_t ncol)
{
for(int i=0;i<nrow;i++) {
for(int j=0;j<ncol;j++) {
free(v[i][j]);
}
free(v[i]);
}
free(v);
}
int64_t getnumlines(const char *fname,const char comment)
{
const int MAXLINESIZE = 10000;
int64_t nlines=0;
char str_line[MAXLINESIZE];
FILE *fp = my_fopen(fname,"rt");
if(fp == NULL) {
return -1;
}
while(1){
if(fgets(str_line, MAXLINESIZE,fp)!=NULL) {
char *c = &str_line[0];
while(*c != '\0' && isspace(*c)) {
c++;
}
if(*c != '\0' && *c !=comment) {
nlines++;
}
} else {
break;
}
}
fclose(fp);
return nlines;
}
int test_all_files_present(const int nfiles, ...)
{
int absent=0;
va_list filenames;
va_start(filenames, nfiles);
XASSERT(nfiles <= 31, "Can only test for 31 files simultaneously. nfiles = %d \n",nfiles);
for(int i=0;i<nfiles;i++) {
const char *f = va_arg(filenames, const char *);
FILE *fp = fopen(f,"r");
if(fp == NULL) {
absent |= 1;
} else {
fclose(fp);
}
absent <<= 1;
}
va_end(filenames);
return absent;
}
int AlmostEqualRelativeAndAbs_float(float A, float B,
const float maxDiff,
const float maxRelDiff)
{
float diff = fabsf(A - B);
if (diff <= maxDiff)
return EXIT_SUCCESS;
A = fabsf(A);
B = fabsf(B);
float largest = (B > A) ? B : A;
if (diff <= largest * maxRelDiff)
return EXIT_SUCCESS;
return EXIT_FAILURE;
}
int AlmostEqualRelativeAndAbs_double(double A, double B,
const double maxDiff,
const double maxRelDiff)
{
double diff = fabs(A - B);
if (diff <= maxDiff)
return EXIT_SUCCESS;
A = fabs(A);
B = fabs(B);
double largest = (B > A) ? B : A;
if (diff <= largest * maxRelDiff)
return EXIT_SUCCESS;
return EXIT_FAILURE;
}
void parallel_cumsum(const int64_t *a, const int64_t N, int64_t *cumsum){
if (N <= 0){
return;  
}
#ifdef _OPENMP
int nthreads = omp_get_max_threads();
#else
int nthreads = 1;
#endif
int64_t min_N_per_thread = 10000;
if(N/min_N_per_thread < nthreads){
nthreads = N/min_N_per_thread;
}
if(nthreads < 1){
nthreads = 1;
}
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
{
#ifdef _OPENMP
int tid = omp_get_thread_num();
#else
int tid = 0;
#endif
int64_t cstart = N*tid/nthreads;
int64_t cend = N*(tid+1)/nthreads;
cumsum[cstart] = cstart > 0 ? a[cstart-1] : 0;
for(int64_t c = cstart+1; c < cend; c++){
cumsum[c] = a[c-1] + cumsum[c-1];
}
#ifdef _OPENMP
#pragma omp barrier
#endif
int64_t offset = 0;
for(int t = 0; t < tid; t++){
offset += cumsum[N*(t+1)/nthreads-1];
}
#ifdef _OPENMP
#pragma omp barrier
#endif
if(offset != 0){
for(int64_t c = cstart; c < cend; c++){
cumsum[c] += offset;
}
}
}
}
