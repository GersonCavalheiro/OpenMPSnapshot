#if !defined(AFX_THREAD_CPY_H__7CA72181_E97E_40DE_9E7E_8A141A6FC68E__INCLUDED_)
#define AFX_THREAD_CPY_H__7CA72181_E97E_40DE_9E7E_8A141A6FC68E__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif 




class thread_cpy : public CWinThread
{
DECLARE_DYNCREATE(thread_cpy)
protected:
thread_cpy();           

public:
private:
long dimensiune;
int thread;
double *freev;
double *freev1;
double **matr1;
double **matr;
long pos;

public:
int Run(void);

public:
thread_cpy(long dim,int threads,double *y,double *y1,double **mat,double **mat1,long *index);
virtual BOOL InitInstance();
virtual int ExitInstance();

protected:
virtual ~thread_cpy();


DECLARE_MESSAGE_MAP();
};



#endif 
