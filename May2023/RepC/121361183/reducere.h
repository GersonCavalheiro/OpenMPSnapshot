
#if !defined(AFX_REDUCERE_H__B1023378_E71C_4766_BCB7_6E3DEEC0A58D__INCLUDED_)
#define AFX_REDUCERE_H__B1023378_E71C_4766_BCB7_6E3DEEC0A58D__INCLUDED_
#include <afxmt.h>
#include <sys/timeb.h>
#include <time.h>

#if _MSC_VER > 1000
#pragma once
#endif 

class reducere  
{
public:
reducere();
virtual ~reducere();

private:
long paralel;
long serial;
void reducere_paralela();
void reducere_seriala();
void timp(LARGE_INTEGER val1,LARGE_INTEGER val2,LARGE_INTEGER val3,LARGE_INTEGER val4);
int* vector;
long dim;
int threads;

};

#endif 
