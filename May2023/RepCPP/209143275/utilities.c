const unsigned long oneMB_double         = 131072;
const unsigned long twoMB_double         = 2*oneMB_double;
const unsigned long threeMB_double       = 3*oneMB_double;
const unsigned long fourMB_double        = 4*oneMB_double;
const unsigned long eightMB_double       = 8*oneMB_double;
const unsigned long sixteenMB_double     = 16*oneMB_double;
const unsigned long thirtytwoMB_double   = 32*oneMB_double;
const unsigned long sixtyfourMB_double   = 64*oneMB_double;

const unsigned long sixtyfourKB_double   = oneMB_double/16; 
const unsigned long oneeighthMB_double   = oneMB_double/8;  
const unsigned long onefourthMB_double   = oneMB_double/4;
const unsigned long halfMB_double        = oneMB_double/2;

#define DEFINE_ARR(an, sz)		double *an = NULL; an = new double [sz]; first_touch_array(an, sz)
#define DEFINE_MTX(an, sz)		double *an = NULL; an = new double [sz*sz]; first_touch_matrix(an, sz)
#define DEFINE_ARR_LD(an, sz)	long double *an = NULL; an = new long double [sz]; first_touch_array_ld(an, sz)
#define DEFINE_MTX_LD(an, sz)	long double *an = NULL; an = new long double [sz*sz]; first_touch_matrix_ld(an, sz)

#define         GREEN           "\e[1;32m"
#define         RED             "\e[1;31m"
#define         NORMAL          "\e[0;0m"

#define 		N  				sixtyfourKB_double
#define 		SMALL_N  		oneeighthMB_double

void 
first_touch_array(double *an, uint sz)
{
assert(an);
#pragma omp parallel for
for(int i=0;i<sz;i++) an[i]=0.; 
}

void 
first_touch_matrix(double *an, uint sz)
{
assert(an);
#pragma omp parallel for
for(int i=0;i<sz;i++) for(int j=0;j<sz;j++)  an[i*sz+j]=0.; 
}

void 
first_touch_array_ld(long double *an, uint sz)
{
#pragma omp parallel for
for(int i=0;i<sz;i++) an[i]=0.; 
assert(an);
}


void 
first_touch_matrix_ld(long double *an, uint sz)
{
assert(an);
#pragma omp parallel for
for(int i=0;i<sz;i++) for(int j=0;j<sz;j++)  an[i*sz+j]=0.; 
}
