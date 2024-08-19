
#if !defined(AFX_GAUSS_H__953CCD66_737A_4480_B76C_95AC40B14FFA__INCLUDED_)
#define AFX_GAUSS_H__953CCD66_737A_4480_B76C_95AC40B14FFA__INCLUDED_
#include <afxmt.h>
#include "bariera.h"
#include "solveGE.h"
#if _MSC_VER > 1000
#pragma once
#endif 

class gauss  
{
public:
gauss();
gauss(long dim,int thread,double **mat,double *x,double *y);
virtual ~gauss();
private:
long *data;
CCriticalSection *crit;
int *counters;
long i;
HANDLE *hthreads;
bariera barrier;
solveGE *threads;
};

#endif 
