#if !defined(AFX_SERIALIZARE_SEM_H__1D897446_F974_4A45_9C86_36E18E745C36__INCLUDED_)
#define AFX_SERIALIZARE_SEM_H__1D897446_F974_4A45_9C86_36E18E745C36__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif 

#include <afxmt.h>


class serializare_sem : public CWinThread
{
DECLARE_DYNCREATE(serializare_sem)
protected:
serializare_sem();           

public:

public:

public:
virtual BOOL InitInstance();
virtual BOOL InitInstance(int *indexare);
virtual int ExitInstance();
serializare_sem(int nr,CSemaphore *semaphore);
int Run();
protected:
virtual ~serializare_sem();


DECLARE_MESSAGE_MAP()
private:
CSemaphore* sem;
int index;
int threads;
};



#endif 
