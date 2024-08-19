#if !defined(AFX_SERIALIZARE_EVENT_H__CA8B1DAC_7830_432E_8E07_C3DFBC1DDDDF__INCLUDED_)
#define AFX_SERIALIZARE_EVENT_H__CA8B1DAC_7830_432E_8E07_C3DFBC1DDDDF__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif 
#include <afxmt.h>



class serializare_event : public CWinThread
{
DECLARE_DYNCREATE(serializare_event)
protected:
serializare_event();           

public:

public:
int Run();

public:
serializare_event(int nr,CEvent *ev,int *indexare);
virtual BOOL InitInstance();
virtual int ExitInstance();

protected:
virtual ~serializare_event();

private:
int index;
CEvent *event;
int threads;
DECLARE_MESSAGE_MAP()
};



#endif 
