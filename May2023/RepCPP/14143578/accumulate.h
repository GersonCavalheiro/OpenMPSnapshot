

#pragma once
#include "config.h"
#include "mylibm.hpp"

namespace dg
{
namespace exblas {
namespace cpu {
#ifndef _WITHOUT_VCL
static inline vcl::Vec8d make_vcl_vec8d( double x, int i){
return vcl::Vec8d(x);
}
static inline vcl::Vec8d make_vcl_vec8d( const double* x, int i){
return vcl::Vec8d().load( x+i);
}
static inline vcl::Vec8d make_vcl_vec8d( double x, int i, int num){
return vcl::Vec8d(x);
}
static inline vcl::Vec8d make_vcl_vec8d( const double* x, int i, int num){
return vcl::Vec8d().load_partial( num, x+i);
}
static inline vcl::Vec8d make_vcl_vec8d( float x, int i){
return vcl::Vec8d((double)x);
}
static inline vcl::Vec8d make_vcl_vec8d( const float* x, int i){
return vcl::Vec8d( x[i], x[i+1], x[i+2], x[i+3], x[i+4], x[i+5], x[i+6], x[i+7]);
}
static inline vcl::Vec8d make_vcl_vec8d( float x, int i, int num){
return vcl::Vec8d((double)x);
}
static inline vcl::Vec8d make_vcl_vec8d( const float* x, int i, int num){
double tmp[8];
for(int j=0; j<num; j++)
tmp[j] = (double)x[i+j];
return vcl::Vec8d().load_partial( num, tmp);
}
#endif
template<class T>
inline double get_element( T x, int i){
return (double)x;
}
template<class T>
inline double get_element( T* x, int i){
return (double)(*(x+i));
}


static inline void AccumulateWord( int64_t *accumulator, int i, int64_t x) {
unsigned char overflow;
int64_t carry = x;
int64_t carrybit;
int64_t oldword = cpu::xadd(accumulator[i], x, overflow);
while(unlikely(overflow)) {
carry = (oldword + carry) >> DIGITS;    
bool s = oldword > 0;
carrybit = (s ? 1ll << KRX : (unsigned long long)(-1ll) << KRX); 

cpu::xadd(accumulator[i], (int64_t) -(carry << DIGITS), overflow);
if(TSAFE && unlikely(s ^ overflow)) {
carrybit *= 2;
}
carry += carrybit;

++i;
if (i >= BIN_COUNT){
return;
}
oldword = cpu::xadd(accumulator[i], carry, overflow);
}
}


static inline void Accumulate( int64_t* accumulator, double x) {
if (x == 0)
return;


int e = cpu::exponent(x);
int exp_word = e / DIGITS;  
int iup = exp_word + F_WORDS;

double xscaled = cpu::myldexp(x, -DIGITS * exp_word);

int i;
for (i = iup; i>=0 && xscaled != 0; --i) { 
double xrounded = cpu::myrint(xscaled);
int64_t xint = cpu::myllrint(xscaled);
AccumulateWord(accumulator, i, xint);

xscaled -= xrounded;
xscaled *= DELTASCALE;
}
}
#ifndef _WITHOUT_VCL
static inline void Accumulate( int64_t* accumulator, vcl::Vec8d x) {
double v[8];
x.store(v);

#if INSTRSET >= 7
_mm256_zeroupper();
#endif
for(unsigned int j = 0; j != 8; ++j) {
exblas::cpu::Accumulate(accumulator, v[j]);
}
}
#endif 

static inline bool Normalize( int64_t *accumulator, int& imin, int& imax) {
int64_t carry_in = accumulator[imin] >> DIGITS;
accumulator[imin] -= carry_in << DIGITS;
int i;
for (i = imin + 1; i < BIN_COUNT; ++i) {
accumulator[i] += carry_in;
int64_t carry_out = accumulator[i] >> DIGITS;    
accumulator[i] -= (carry_out << DIGITS);
carry_in = carry_out;
}
imax = i - 1;

accumulator[imax] += carry_in << DIGITS;

return carry_in < 0;
}




static inline double Round( int64_t * accumulator) {
int imin = IMIN;
int imax = IMAX;
bool negative = Normalize(accumulator, imin, imax);

int i;
for(i = imax; i >= imin && accumulator[i] == 0; --i) {
}
if (negative) {
for(; i >= imin && (accumulator[i] & ((1ll << DIGITS) - 1)) == ((1ll << DIGITS) - 1); --i) {
}
}
if (i < 0) {
return 0.0;
}

int64_t hiword = negative ? ((1ll << DIGITS) - 1) - accumulator[i] : accumulator[i];
double rounded = (double)hiword;
double hi = ldexp(rounded, (i - F_WORDS) * DIGITS);
if (i == 0) {
return negative ? -hi : hi;  
}
hiword -= std::llrint(rounded);
double mid = ldexp((double) hiword, (i - F_WORDS) * DIGITS);

int64_t sticky = 0;
for (int j = imin; j != i - 1; ++j) {
sticky |= negative ? ((1ll << DIGITS) - accumulator[j]) : accumulator[j];
}

int64_t loword = negative ? ((1ll << DIGITS) - accumulator[i - 1]) : accumulator[i - 1];
loword |= !!sticky;
double lo = ldexp((double) loword, (i - 1 - F_WORDS) * DIGITS);

if (mid != 0) {
lo = cpu::OddRoundSumNonnegative(mid, lo);
}
hi = hi + lo;
return negative ? -hi : hi;
}

}
} 
} 
