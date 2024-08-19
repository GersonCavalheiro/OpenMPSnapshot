
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <gmp.h>
#include <omp.h>
double binomial_coeff(double n, double r)
{
double i, j, s=1.0;
if (r > n) return 0;
for (i=n; i>(n-r); i-=1.0)
s *= i;
for (j=1.0; j<=r; j+=1.0)
s /= j;
return s;
}
void mpf_binomial_coeff(mpf_ptr b, mpf_srcptr n, mpf_srcptr r)
{
mpf_t i, j, k, s;
mpf_set_ui(b, 0);
if (mpf_cmp(n, r) < 0) return;
mpf_init_set(i, n);
mpf_init_set_ui(s, 1L);
mpf_init(k);
mpf_sub(k, n, r);
for (; mpf_cmp(i, k)>0; mpf_sub_ui(i, i, 1L))
mpf_mul(s, s, i);
mpf_init_set_ui(j, 1L);
mpf_add_ui(k, r, 1L);
for (; mpf_cmp(j, k)<0; mpf_add_ui(j, j, 1L))
mpf_div(s, s, j);
mpf_ceil(s, s);
mpf_set(b, s);
}
void mpz_binomial_coeff_2(mpz_ptr b, int n, int r)
{
mpz_t s;
mpz_set_ui(b, 0);
if (n < r) return;
mpz_init_set_ui(s, 1L);
for (int i=n; i>(n-r); i--)
mpz_mul_ui(s, s, i);
for (int j=1; j<=r; j++)
mpz_div_ui(s, s, j);
mpz_set(b, s);
}
void machin_like(mpf_ptr sum, int N)
{
mpf_t n, t;
mpf_t pow_2;
mpf_t ONE;
mpz_t b;
mpf_inits(n, t, pow_2, ONE, NULL);
mpz_init(b);
for (int k = 1; k < N; k++)
{
mpf_set_ui(ONE, 1UL);
mpf_mul_2exp(pow_2, ONE, k);            
mpz_bin_uiui(b, 2*k, k);                
mpf_set_z(n, b);
mpf_div(t, pow_2, n);                   
mpf_add(sum, sum, t);               
}
mpf_add(sum, sum, sum);
mpf_sub_ui(sum, sum, 2L);
mpz_clear(b);
mpf_clears(n, t, pow_2, ONE, NULL);
}
void arctan_of_one(mpf_ptr sum, int N)
{
mpf_t m, n;
mpf_inits(m, n, NULL);
int MUL = 1000000;
for (int j = 0; j < N*MUL; j += MUL)
{
for (int i = j; i < j+MUL; i++)
{
mpf_set_si(m, -1L);                 
mpf_div_ui(m, m, i*4+3);            
mpf_add(sum, sum, m);               
mpf_set_ui(n, 1UL);
mpf_div_ui(n, n, i*4+1);            
mpf_add(sum, sum, n);               
}
putchar('.');
}
printf("\n");
mpf_mul_ui(sum, sum, 4L);               
mpf_clears(m, n, NULL);
}
void lucas_formula(mpf_ptr sum, int N)
{
mpf_t n, t, four_k;
mpf_t ONE, FOUR;
mpf_init(n);
mpf_init(t);
mpf_init(four_k);
mpf_init_set_ui(ONE, 1L);
mpf_init_set_ui(FOUR, 4L);
mpf_mul_ui(sum, sum, 16L);
for (int k = 0; k < N; k++)
{
mpf_set(n, ONE);                
mpf_mul_ui(four_k, FOUR, k);    
mpf_add_ui(t, four_k, 1L);      
mpf_div(n, n, t);               
mpf_div(n, n, t);               
mpf_add_ui(t, four_k, 3L);      
mpf_div(n, n, t);               
mpf_div(n, n, t);               
mpf_add_ui(t, four_k, 5L);      
mpf_div(n, n, t);               
mpf_div(n, n, t);               
gmp_printf("%0.Ff\n", n);
mpf_add(sum, sum, n);
}
gmp_printf("%0.Ff\n", sum);
}
void gosper_formula(mpf_ptr sum, int N)
{
mpf_t pow_2, _25k_sub3;
mpf_t n, t;
mpz_t b;
mpz_init(b);
mpf_set_si(sum, -6L);                       
mpf_inits(pow_2, _25k_sub3, n, t, NULL);
for (int k=1; k<N; k++)
{
mpf_set_ui(_25k_sub3, k*25L-3L);        
mpz_bin_uiui(b, 3*k, k);                
mpf_set_z(n, b);                        
mpf_div_2exp(t, _25k_sub3, k-1);        
mpf_div(t, t, n);                       
mpf_add(sum, sum, t);                   
printf("%8d\r", k);
}
mpf_clears(pow_2, t, _25k_sub3, n, NULL);
mpz_clear(b);
}
void bbp_formula(mpf_ptr sum)
{
mpf_t N, T, EPSILON, ABS_N;
mpf_inits(N, T, EPSILON, ABS_N, NULL);
mpf_set_ui(EPSILON, 1);
int precision = (int)(mpf_get_default_prec());
mpf_div_2exp(EPSILON, EPSILON, precision + (precision >> 2));
bool bBreak = false;
int k = 0;
for (; !bBreak; k++)
{
mpf_set_ui(T, 4UL);                     
mpf_div_ui(T, T, 8*k+1);                
mpf_set(N, T);                          
mpf_set_ui(T, 2UL);                     
mpf_div_ui(T, T, 8*k+4);                
mpf_sub(N, N, T);                       
mpf_set_ui(T, 1UL);                     
mpf_div_ui(T, T, 8*k+5);                
mpf_sub(N, N, T);                       
mpf_set_ui(T, 1UL);                     
mpf_div_ui(T, T, 8*k+6);                
mpf_sub(N, N, T);                       
mpf_div_2exp(N, N, 4*k);                
mpf_add(sum, sum, N);                   
mpf_abs(ABS_N, N);
bBreak = (mpf_cmp(ABS_N, EPSILON) < 0);
}
printf("iterations: %d\n", k);
mpf_clears(N, T, EPSILON, ABS_N, NULL);
}
void chudnovsky_formula(mpf_ptr Sum)
{
mpf_t C1, C2, M, N, D, EPSILON, ABS_N, LOG10_BASE2;
mpz_t L, M1, M2, X;
mpf_inits(C1, C2, M, N, D, EPSILON, ABS_N, NULL);
mpz_inits(L, M1, M2, X, NULL);
mpf_set_ui(EPSILON, 1);
mpf_init_set_d(LOG10_BASE2, log(10.0) / log(2.0));
int precision = (int)(mpf_get_default_prec());
mpf_div_2exp(EPSILON, EPSILON, precision + (precision >> 2));
mpf_div(EPSILON, EPSILON, LOG10_BASE2);
gmp_printf("EPSILON = %.80Fg\n\n", EPSILON);
bool bBreak = false;
unsigned int k = 0;
for (; !bBreak; k++)
{
mpz_fac_ui(M1, 6*k);                    
mpz_fac_ui(M2, 3*k);					
mpz_div(M1, M1, M2);                    
mpz_fac_ui(M2, k);				        
mpz_pow_ui(M2, M2, 3);                  
mpz_div(M1, M1, M2);                    
mpz_set_ui(L, 545140134);               
mpz_mul_ui(L, L, k);                    
mpz_add_ui(L, L, 13591409);             
mpz_set_si(X, -640320);                 
mpz_pow_ui(X, X, 3*k);                  
mpf_set_z(N, L);						
mpf_set_z(D, X);						
mpf_div(N, N, D);						
mpf_abs(ABS_N, N);
bBreak = (mpf_cmp(ABS_N, EPSILON) < 0);
mpf_set_z(M, M1);
mpf_mul(N, M, N);
mpf_add(Sum, Sum, N);
}
printf("iterations: %d\n", k);
mpf_set_ui(C1, 426880);
mpf_sqrt_ui(C2, 10005);
mpf_mul(C1, C2, C1);
mpf_div(Sum, C1, Sum);
mpf_clears(C1, C2, M, N, D, ABS_N, EPSILON, LOG10_BASE2, NULL);
mpz_clears(L, M1, M2, X, NULL);
}
void gauss_legendre_algorithm(mpf_ptr sum, int N)
{
mpf_t a0, b0, t0, p0;
mpf_t a1, b1, t1;
int n_digits = 80;
mpf_inits(a0, b0, p0, t0, NULL);
mpf_inits(a1, b1, t1, NULL);
mpf_set_ui(a0, 1L);
mpf_set_ui(p0, 1L);
mpf_sqrt_ui(b0, 2L);
mpf_ui_div(b0, 1L, b0);
mpf_init_set_d(t0, 0.25d);
for (int i = 0; i < N; i++)
{
mpf_add(a1, a0, b0);                
mpf_div_ui(a1, a1, 2UL);            
mpf_mul(b1, a0, b0);                
mpf_sqrt(b1, b1);                   
mpf_sub(t1, a0, a1);                
mpf_pow_ui(t1, t1, 2UL);            
mpf_mul(t1, p0, t1);                
mpf_sub(t1, t0, t1);                
mpf_mul_ui(p0, p0, 2UL);            
gmp_printf ("a1: %.*Ff with %d digits\n", n_digits, a1, n_digits);
gmp_printf ("b1: %.*Ff with %d digits\n", n_digits, b1, n_digits);
gmp_printf ("t1: %.*Ff with %d digits\n", n_digits, t1, n_digits);
gmp_printf ("p0: %.*Ff with %d digits\n\n", n_digits, p0, n_digits);
mpf_set(a0, a1);
mpf_set(b0, b1);
mpf_set(t0, t1);
}
mpf_add(sum, a1, b1);               
mpf_pow_ui(sum, sum, 2UL);          
mpf_div(sum, sum, t1);              
mpf_div_ui(sum, sum, 4UL);          
gmp_printf ("pi ~= %.*Ff with %d digits\n", n_digits, sum, n_digits);
mpf_clears(a0, b0, p0, t0, a1, b1, t1, NULL);
}    
int main(int argc, char *argv[])
{
FILE *fs;
int prec, iter = 11;
mpf_t sum;
clock_t start, stop;
if (argc != 2)
{
prec = 1024;
}
else
{
prec = atoi(argv[1]);
}
printf("GMP version: %s\n", gmp_version);
mpf_set_default_prec(prec);
mpf_init(sum);
start = clock();
bbp_formula(sum);
stop = clock();
printf("\n\nElapsed time: %6.2f seconds\n",  
(float)(stop-start)/(float)CLOCKS_PER_SEC ); 
if ((fs = fopen("gmp_pi.txt", "w+t")) != NULL)
{
char buffer[1048576];
gmp_sprintf(buffer, "%0.Ff\n", sum);
fprintf(fs, "%s", buffer);
fclose(fs);
}
mpf_clear(sum);
return EXIT_SUCCESS;
}
