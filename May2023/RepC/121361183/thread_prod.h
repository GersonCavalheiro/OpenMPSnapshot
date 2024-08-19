#if !defined(AFX_THREAD_PROD_H__28096C5E_AFC4_41A1_B723_713046904447__INCLUDED_)
#define AFX_THREAD_PROD_H__28096C5E_AFC4_41A1_B723_713046904447__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif 




class thread_prod : public CWinThread
{
DECLARE_DYNCREATE(thread_prod)
protected:
thread_prod();           

public:

public:
int Run();
thread_prod(long dimensiune,int proc,double **mat0,double **mat1,double **mat2,int *indexare);
public:
virtual BOOL InitInstance();
virtual int ExitInstance();

protected:
virtual ~thread_prod();


DECLARE_MESSAGE_MAP()
private:
long dim;
int threads;
double **mata;
double **matb;
double **matc;
int index;
};



#endif 
