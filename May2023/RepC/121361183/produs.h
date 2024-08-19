
#if !defined(AFX_PRODUS_H__43EE0EDF_04AC_4DC6_9083_62461D26C9D6__INCLUDED_)
#define AFX_PRODUS_H__43EE0EDF_04AC_4DC6_9083_62461D26C9D6__INCLUDED_

#include "stdafx.h"
#include <sys/timeb.h>
#include <time.h>
#include <fstream.h>

#if _MSC_VER > 1000
#pragma once
#endif 

class produs  
{
public:

produs();
virtual ~produs();

private:
void produs::timpp(LARGE_INTEGER val1,LARGE_INTEGER val2);
void produs::timps(LARGE_INTEGER val1,LARGE_INTEGER val2);
void verificare_calcul(double **mat1,double **mat2);
void calcul_ser(double **matc);
void calcul_par(double **matc);
int threaduri[6];
ofstream *fp;
HANDLE *hthreads;
double **mata;
double **matb;
double **matc1;
double **matc2;
long dim;
int threads;
int *ordine;
int index;
};

#endif 
