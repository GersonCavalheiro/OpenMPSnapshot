#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void costa_pstran(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void costa_pdtran(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void costa_pstran_(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void costa_pdtran_(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void costa_pstran__(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void costa_pdtran__(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void COSTA_PSTRAN(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void COSTA_PDTRAN(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );
#ifdef __cplusplus
}
#endif
