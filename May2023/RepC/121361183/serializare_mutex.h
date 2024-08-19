#if !defined(AFX_SERIALIZARE_MUTEX_H__EA6E98C3_D170_4473_8FE1_1DC806878B05__INCLUDED_)
#define AFX_SERIALIZARE_MUTEX_H__EA6E98C3_D170_4473_8FE1_1DC806878B05__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif 

#include <afxmt.h>


class serializare_mutex : public CWinThread
{
DECLARE_DYNCREATE(serializare_mutex)
protected:
serializare_mutex();           

public:

public:

public:
virtual BOOL InitInstance();
serializare_mutex(int nr,CMutex *mtx,int *indexare);
virtual int ExitInstance();
int Run();

protected:
virtual ~serializare_mutex();

DECLARE_MESSAGE_MAP()
private:
CMutex* mutexuri;
int threads;
int index;
};



#endif 
