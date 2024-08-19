
#if !defined(AFX_BENCH_H__065D5516_0201_47E4_8349_C137B0DADDCE__INCLUDED_)
#define AFX_BENCH_H__065D5516_0201_47E4_8349_C137B0DADDCE__INCLUDED_
#include <fstream.h>
#include <sys/timeb.h>
#include <time.h>
#include "stdafx.h"
#include "thread_cpy.h"
#include "gauss.h"

#if _MSC_VER > 1000
#pragma once
#endif 

class bench  
{
public:
bench();
virtual ~bench();
double **mat;
double *x;
double *rez;
double *y;
double **mat1;
double *y1;
private:
long *ordine;
void timpp(LARGE_INTEGER val1,LARGE_INTEGER val2);
void timps(LARGE_INTEGER val1,LARGE_INTEGER val2);
void gauss_ser(long dim1,double **matr,double *sol,double *freev);
int threaduri[6];
ofstream *fp;
long dim;
int threads;
HANDLE *hthreads;
thread_cpy *thread;
gauss *calcul;
};

#endif 
