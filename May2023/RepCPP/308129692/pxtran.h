#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void pstran(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void pdtran(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void pstran_(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void pdtran_(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void pstran__(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void pdtran__(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void PSTRAN(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void PDTRAN(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );
#ifdef __cplusplus
}
#endif
