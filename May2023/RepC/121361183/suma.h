#if !defined(AFX_SUMA_H__FA7A27C5_BBB7_4838_9D10_EB3C3702D010__INCLUDED_)
#define AFX_SUMA_H__FA7A27C5_BBB7_4838_9D10_EB3C3702D010__INCLUDED_
#include <afxmt.h>
#if _MSC_VER > 1000
#pragma once
#endif 




class suma : public CWinThread
{
DECLARE_DYNCREATE(suma)
protected:
suma();           

public:

public:
int Run();

public:
suma(long dimensiune,int nrthreads,int *vectori,CMutex *mutexuri,long *paralel,int *poz);
virtual BOOL InitInstance();
virtual int ExitInstance();

protected:
virtual ~suma();


DECLARE_MESSAGE_MAP()
private:
long  *rezultat;
long dim;
int threads;
int *vector;
CMutex *mutex;
int index;
};



#endif 
