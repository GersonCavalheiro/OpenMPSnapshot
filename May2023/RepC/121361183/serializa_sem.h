#if !defined(AFX_SERIALIZA_SEM_H__02491FCA_32FD_4976_B1B3_6B66BD8C69DC__INCLUDED_)
#define AFX_SERIALIZA_SEM_H__02491FCA_32FD_4976_B1B3_6B66BD8C69DC__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif 

#include <afxmt.h>


class serializa_sem : public CWinThread
{
DECLARE_DYNCREATE(serializa_sem)
protected:
serializa_sem();           
serializa

public:

public:

public:
virtual BOOL InitInstance();
virtual int ExitInstance();

protected:
virtual ~serializa_sem();


DECLARE_MESSAGE_MAP()
};



#endif 
