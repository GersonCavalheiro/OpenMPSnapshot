#ifndef __QRCA_KERNELS_H__
#define __QRCA_KERNELS_H__
#if USE_NODEPS
#pragma omp task 
#else
#pragma omp task inout(  Uh[0;its1*its2], Dh[0;its1*its2]) output( Sh[0;ibs*its1]) priority(2)
#endif
void NoFLA_Compute_td_QR_var31a( int ibs, int its1, int its2, int skip, 
double * Uh, 
double * Dh, 
double * Th,
double * Sh);
#if USE_NODEPS
#pragma omp task 
#else
#pragma omp task inout(  Uh[0;its1*its2], Dh[0;its1*its2]) output( Sh[0;ibs*its1]) priority(2)
#endif
void NoFLA_Compute_td_QR_var31b( int ibs, int its1, int its2, int skip, 
double * Uh, 
double * Dh, 
double * Th,
double * Sh);
#if USE_NODEPS
#pragma omp task 
#else
#pragma omp task input( Sh[0;ibs*its1] ) inout( Fh[0;its1*its2], Gh[0;its1*its2])
#endif
void NoFLA_Apply_td_QT_var31a( int ibs, int its1, int its2, int skip, 
double * Dh, 
double * Sh, 
double * Fh, 
double * Gh);
#endif 
