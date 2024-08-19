#if !defined(AFX_SERIALIZARE_BARIERA_H__1022744A_0EDE_4F4E_A0B7_28AC6B28B018__INCLUDED_)
#define AFX_SERIALIZARE_BARIERA_H__1022744A_0EDE_4F4E_A0B7_28AC6B28B018__INCLUDED_
#include "bariera.h"
#if _MSC_VER > 1000
#pragma once
#endif 




class serializare_bariera : public CWinThread
{
DECLARE_DYNCREATE(serializare_bariera)
protected:
serializare_bariera();           

public:

public:

public:
virtual BOOL InitInstance();
serializare_bariera(int nr,CSemaphore *mtx,int *indexare,bariera *bar);
virtual int ExitInstance();
int Run();

protected:
virtual ~serializare_bariera();

private:
CSemaphore* sem;
int threads;
int index;
bariera *barrier;
DECLARE_MESSAGE_MAP()
};



#endif 
