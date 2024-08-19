#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void costa_pctranu(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void costa_pztranu(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void costa_pctranu_(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void costa_pztranu_(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void costa_pctranu__(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void costa_pztranu__(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );

void COSTA_PCTRANU(const int *m , const int *n , 
float *alpha , const float *a , 
const int *ia , const int *ja , 
const int *desca , 
const float *beta , float *c , 
const int *ic , const int *jc ,
const int *descc );

void COSTA_PZTRANU(const int *m , const int *n , 
double *alpha , const double *a , 
const int *ia , const int *ja , 
const int *desca , 
const double *beta , double *c , 
const int *ic , const int *jc ,
const int *descc );
#ifdef __cplusplus
}
#endif
