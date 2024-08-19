#if !defined(AFX_SERIALIZARE_CRITICALS_H__C57703C9_9F4B_4537_9DF2_766EA9C58872__INCLUDED_)
#define AFX_SERIALIZARE_CRITICALS_H__C57703C9_9F4B_4537_9DF2_766EA9C58872__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif 

#include <afxmt.h>


class serializare_criticals : public CWinThread
{
DECLARE_DYNCREATE(serializare_criticals)
protected:
serializare_criticals();           

public:

public:
int Run();

public:
serializare_criticals(int nr,CCriticalSection *crit,int *indexare);
virtual BOOL InitInstance();
virtual int ExitInstance();

protected:
virtual ~serializare_criticals();


DECLARE_MESSAGE_MAP()
private:
int index;
int threads;
CCriticalSection *critical;
};



#endif 
