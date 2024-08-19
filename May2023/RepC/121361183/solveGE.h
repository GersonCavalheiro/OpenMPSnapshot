#if !defined(AFX_SOLVEGE_H__F456FCF5_F5C0_41BD_BDC5_C8ABAA7267DB__INCLUDED_)
#define AFX_SOLVEGE_H__F456FCF5_F5C0_41BD_BDC5_C8ABAA7267DB__INCLUDED_
#include <afxmt.h>
#include "bariera.h"
#if _MSC_VER > 1000
#pragma once
#endif 




class solveGE : public CWinThread
{
DECLARE_DYNCREATE(solveGE)
protected:
solveGE();           
CCriticalSection *crit;
double **mat;
double *y;
int who;
long N;
int P;
int *counters;
bariera *barrier;
public:

public:
int Run(void);

public:
solveGE(double **matrix,double *yterm,long dim,int dim_threads,CCriticalSection *critical,int *count,bariera *bar,int whoindex);
virtual BOOL InitInstance();
virtual int ExitInstance();

protected:
virtual ~solveGE();


DECLARE_MESSAGE_MAP()
private:
long s;				
long nr;			
long lastrow;		
long proccount;		
long proccount1;	
long replay;		
long i,j,k,l;		
};



#endif 
