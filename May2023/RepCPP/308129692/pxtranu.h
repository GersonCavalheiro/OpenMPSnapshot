#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void pctranu(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void pztranu(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );


void pctranu_(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void pztranu_(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void pctranu__(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void pztranu__(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void PCTRANU(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void PZTRANU(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

#ifdef __cplusplus
}
#endif
