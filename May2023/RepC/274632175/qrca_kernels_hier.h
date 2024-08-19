#ifndef __QRCA_KERNELS_H__
#define __QRCA_KERNELS_H__
#pragma omp task
void split_dlarfb_hier_task(int cm, int cn, double *C, int ldim_C, double *U, int ldim_U, int dimt, double *T, int ldim_T, int fm, double *F, double *work);
#pragma omp task
void NoFLA_QR_td_unb_var1_updateS( int mn_U, int m_D, 
double * buff_D, int ldim_D,
double * buff_t,
double * buff_S, int ldim_S ); 
#pragma omp task input( Sh[0;t*m_A] ) inout( Ch[0;m_A*n_A]) 
void dlarfb_task_hier( int t, int m_A, int n_A, int n_C, int skip, 
double * Uh, 
double * Sh, 
double * Ch);
#pragma omp task inout(  Uh[0;its1*its1], Dh[0;its1*its1]) output( Sh[0;ibs*its1]) priority(1)
void NoFLA_Compute_td_QR_var31a_hier( int ibs, int its1, int its2, int skip, 
double * Uh, 
double * Dh, 
double * Th,
double * Sh);
#pragma omp task input( Sh[0;ibs*its1] ) inout( Fh[0;its1*its1], Gh[0;its1*its1])
void NoFLA_Apply_td_QT_var31a_hier( int ibs, int its1, int its2, int skip, 
double * Dh, 
double * Sh, 
double * Fh, 
double * Gh);
#endif 
