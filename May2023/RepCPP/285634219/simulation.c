#include "rsbench.h"


void run_event_based_simulation(Input input, SimulationData data, unsigned long * vhash_result)
{
printf("Beginning event based simulation ...\n");
unsigned long verification = 0;

#pragma omp target data map(to:data.n_poles[:data.length_n_poles])\
map(to:data.n_windows[:data.length_n_windows])\
map(to:data.poles[:data.length_poles])\
map(to:data.windows[:data.length_windows])\
map(to:data.pseudo_K0RS[:data.length_pseudo_K0RS])\
map(to:data.num_nucs[:data.length_num_nucs])\
map(to:data.mats[:data.length_mats])\
map(to:data.concs[:data.length_concs])\
map(to:data.max_num_nucs)\
map(to:data.max_num_poles)\
map(to:data.max_num_windows)\
map(tofrom:verification)
{
double start = get_time();

#pragma omp target teams distribute parallel for reduction(+:verification)
for( int i = 0; i < input.lookups; i++ )
{
uint64_t seed = STARTING_SEED;  

seed = fast_forward_LCG(seed, 2*i);

double E = LCG_random_double(&seed);
int mat  = pick_mat(&seed);

double macro_xs[4] = {0};

calculate_macro_xs( macro_xs, mat, E, input, data.num_nucs, data.mats,
data.max_num_nucs, data.concs, data.n_windows,
data.pseudo_K0RS, data.windows, data.poles,
data.max_num_windows, data.max_num_poles );

double max = -DBL_MAX;
int max_idx = 0;
for(int x = 0; x < 4; x++ )
{
if( macro_xs[x] > max )
{
max = macro_xs[x];
max_idx = x;
}
}
verification += max_idx+1;
}

double stop = get_time();
printf("Kernel initialization, compilation, and launch took %.2lf seconds.\n", stop-start);
}

*vhash_result = verification;
}

void calculate_macro_xs(double * macro_xs, int mat, double E, Input input,
int * num_nucs, int * mats, int max_num_nucs,
double * concs, int * n_windows, double * pseudo_K0Rs,
Window * windows, Pole * poles, int max_num_windows, int max_num_poles ) 
{
for( int i = 0; i < 4; i++ )
macro_xs[i] = 0;

for( int i = 0; i < num_nucs[mat]; i++ )
{
double micro_xs[4];
int nuc = mats[mat * max_num_nucs + i];

if( input.doppler == 1 )
calculate_micro_xs_doppler( micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);
else
calculate_micro_xs( micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);

for( int j = 0; j < 4; j++ )
{
macro_xs[j] += micro_xs[j] * concs[mat * max_num_nucs + i];
}

}


}

void calculate_micro_xs(double * micro_xs, int nuc, double E, Input input,
int * n_windows, double * pseudo_K0RS, Window * windows,
Pole * poles, int max_num_windows, int max_num_poles)
{
double sigT;
double sigA;
double sigF;
double sigE;

double spacing = 1.0 / n_windows[nuc];
int window = (int) ( E / spacing );
if( window == n_windows[nuc] )
window--;

RSComplex sigTfactors[4]; 
calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors );

Window w = windows[nuc * max_num_windows + window];
sigT = E * w.T;
sigA = E * w.A;
sigF = E * w.F;

for( int i = w.start; i < w.end; i++ )
{
RSComplex PSIIKI;
RSComplex CDUM;
Pole pole = poles[nuc * max_num_poles + i];
RSComplex t1 = {0, 1};
RSComplex t2 = {sqrt(E), 0 };
PSIIKI = c_div( t1 , c_sub(pole.MP_EA,t2) );
RSComplex E_c = {E, 0};
CDUM = c_div(PSIIKI, E_c);
sigT += (c_mul(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value])) ).r;
sigA += (c_mul( pole.MP_RA, CDUM)).r;
sigF += (c_mul(pole.MP_RF, CDUM)).r;
}

sigE = sigT - sigA;

micro_xs[0] = sigT;
micro_xs[1] = sigA;
micro_xs[2] = sigF;
micro_xs[3] = sigE;
}

void calculate_micro_xs_doppler(double * micro_xs, int nuc, double E,
Input input, int * n_windows, double * pseudo_K0RS,
Window * windows, Pole * poles, int max_num_windows, int max_num_poles )
{
double sigT;
double sigA;
double sigF;
double sigE;

double spacing = 1.0 / n_windows[nuc];
int window = (int) ( E / spacing );
if( window == n_windows[nuc] )
window--;

RSComplex sigTfactors[4]; 
calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors );

Window w = windows[nuc * max_num_windows + window];
sigT = E * w.T;
sigA = E * w.A;
sigF = E * w.F;

double dopp = 0.5;

for( int i = w.start; i < w.end; i++ )
{
Pole pole = poles[nuc * max_num_poles + i];

RSComplex E_c = {E, 0};
RSComplex dopp_c = {dopp, 0};
RSComplex Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);

RSComplex faddeeva = fast_nuclear_W( Z );

sigT += (c_mul( pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value]) )).r;
sigA += (c_mul( pole.MP_RA , faddeeva)).r;
sigF += (c_mul( pole.MP_RF , faddeeva)).r;
}

sigE = sigT - sigA;

micro_xs[0] = sigT;
micro_xs[1] = sigA;
micro_xs[2] = sigF;
micro_xs[3] = sigE;
}

int pick_mat( uint64_t * seed )
{

double dist[12];
dist[0]  = 0.140;  
dist[1]  = 0.052;  
dist[2]  = 0.275;  
dist[3]  = 0.134;  
dist[4]  = 0.154;  
dist[5]  = 0.064;  
dist[6]  = 0.066;  
dist[7]  = 0.055;  
dist[8]  = 0.008;  
dist[9]  = 0.015;  
dist[10] = 0.025;  
dist[11] = 0.013;  

double roll = LCG_random_double(seed);

for( int i = 0; i < 12; i++ )
{
double running = 0;
for( int j = i; j > 0; j-- )
running += dist[j];
if( roll < running )
return i;
}

return 0;
}

void calculate_sig_T( int nuc, double E, Input input, double * pseudo_K0RS, RSComplex * sigTfactors )
{
double phi;

for( int i = 0; i < 4; i++ )
{
phi = pseudo_K0RS[nuc * input.numL + i] * sqrt(E);

if( i == 1 )
phi -= - atan( phi );
else if( i == 2 )
phi -= atan( 3.0 * phi / (3.0 - phi*phi));
else if( i == 3 )
phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));

phi *= 2.0;

sigTfactors[i].r = cos(phi);
sigTfactors[i].i = -sin(phi);
}
}

RSComplex fast_nuclear_W( RSComplex Z )
{
if( c_abs(Z) < 6.0 )
{
RSComplex prefactor = {0, 8.124330e+01};
double an[10] = {
2.758402e-01,
2.245740e-01,
1.594149e-01,
9.866577e-02,
5.324414e-02,
2.505215e-02,
1.027747e-02,
3.676164e-03,
1.146494e-03,
3.117570e-04
};
double neg_1n[10] = {
-1.0,
1.0,
-1.0,
1.0,
-1.0,
1.0,
-1.0,
1.0,
-1.0,
1.0
};

double denominator_left[10] = {
9.869604e+00,
3.947842e+01,
8.882644e+01,
1.579137e+02,
2.467401e+02,
3.553058e+02,
4.836106e+02,
6.316547e+02,
7.994380e+02,
9.869604e+02
};

RSComplex t1 = {0, 12};
RSComplex t2 = {12, 0};
RSComplex i = {0,1};
RSComplex one = {1, 0};
RSComplex W = c_div(c_mul(i, ( c_sub(one, fast_cexp(c_mul(t1, Z))) )) , c_mul(t2, Z));
RSComplex sum = {0,0};
for( int n = 0; n < 10; n++ )
{
RSComplex t3 = {neg_1n[n], 0};
RSComplex top = c_sub(c_mul(t3, fast_cexp(c_mul(t1, Z))), one);
RSComplex t4 = {denominator_left[n], 0};
RSComplex t5 = {144, 0};
RSComplex bot = c_sub(t4, c_mul(t5,c_mul(Z,Z)));
RSComplex t6 = {an[n], 0};
sum = c_add(sum, c_mul(t6, c_div(top,bot)));
}
W = c_add(W, c_mul(prefactor, c_mul(Z, sum)));
return W;
}
else
{
RSComplex a = {0.512424224754768462984202823134979415014943561548661637413182,0};
RSComplex b = {0.275255128608410950901357962647054304017026259671664935783653, 0};
RSComplex c = {0.051765358792987823963876628425793170829107067780337219430904, 0};
RSComplex d = {2.724744871391589049098642037352945695982973740328335064216346, 0};

RSComplex i = {0,1};
RSComplex Z2 = c_mul(Z, Z);
RSComplex W = c_mul(c_mul(Z,i), (c_add(c_div(a,(c_sub(Z2, b))) , c_div(c,(c_sub(Z2, d))))));

return W;
}
}

double LCG_random_double(uint64_t * seed)
{
const uint64_t m = 9223372036854775808ULL; 
const uint64_t a = 2806196910506780709ULL;
const uint64_t c = 1ULL;
*seed = (a * (*seed) + c) % m;
return (double) (*seed) / (double) m;
}  

uint64_t LCG_random_int(uint64_t * seed)
{
const uint64_t m = 9223372036854775808ULL; 
const uint64_t a = 2806196910506780709ULL;
const uint64_t c = 1ULL;
*seed = (a * (*seed) + c) % m;
return *seed;
}  

uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
const uint64_t m = 9223372036854775808ULL; 
uint64_t a = 2806196910506780709ULL;
uint64_t c = 1ULL;

n = n % m;

uint64_t a_new = 1;
uint64_t c_new = 0;

while(n > 0) 
{
if(n & 1)
{
a_new *= a;
c_new = c_new * a + c;
}
c *= (a + 1);
a *= a;

n >>= 1;
}

return (a_new * seed + c_new) % m;
}


RSComplex c_add( RSComplex A, RSComplex B)
{
RSComplex C;
C.r = A.r + B.r;
C.i = A.i + B.i;
return C;
}

RSComplex c_sub( RSComplex A, RSComplex B)
{
RSComplex C;
C.r = A.r - B.r;
C.i = A.i - B.i;
return C;
}

RSComplex c_mul( RSComplex A, RSComplex B)
{
double a = A.r;
double b = A.i;
double c = B.r;
double d = B.i;
RSComplex C;
C.r = (a*c) - (b*d);
C.i = (a*d) + (b*c);
return C;
}

RSComplex c_div( RSComplex A, RSComplex B)
{
double a = A.r;
double b = A.i;
double c = B.r;
double d = B.i;
RSComplex C;
double denom = c*c + d*d;
C.r = ( (a*c) + (b*d) ) / denom;
C.i = ( (b*c) - (a*d) ) / denom;
return C;
}

double c_abs( RSComplex A)
{
return sqrt(A.r*A.r + A.i * A.i);
}


double fast_exp(double x)
{
x = 1.0 + x * 0.000244140625;
x *= x; x *= x; x *= x; x *= x;
x *= x; x *= x; x *= x; x *= x;
x *= x; x *= x; x *= x; x *= x;
return x;
}

RSComplex fast_cexp( RSComplex z )
{
double x = z.r;
double y = z.i;

double t1 = fast_exp(x);
double t2 = cos(y);
double t3 = sin(y);
RSComplex t4 = {t2, t3};
RSComplex t5 = {t1, 0};
RSComplex result = c_mul(t5, (t4));
return result;
}  
