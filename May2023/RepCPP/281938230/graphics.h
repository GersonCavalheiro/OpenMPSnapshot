

#pragma once

#include <easyx.h>



#define SHOWCONSOLE		1		
#define NOCLOSE			2		
#define NOMINIMIZE		4		
#define EW_SHOWCONSOLE	1		
#define EW_NOCLOSE		2		
#define EW_NOMINIMIZE	4		
#define EW_DBLCLKS		8		



#define	NULL_FILL			BS_NULL
#define	EMPTY_FILL			BS_NULL
#define	SOLID_FILL			BS_SOLID
#define	BDIAGONAL_FILL		BS_HATCHED, HS_BDIAGONAL					
#define CROSS_FILL			BS_HATCHED, HS_CROSS						
#define DIAGCROSS_FILL		BS_HATCHED, HS_DIAGCROSS					
#define DOT_FILL			(BYTE*)"\x80\x00\x08\x00\x80\x00\x08\x00"	
#define FDIAGONAL_FILL		BS_HATCHED, HS_FDIAGONAL					
#define HORIZONTAL_FILL		BS_HATCHED, HS_HORIZONTAL					
#define VERTICAL_FILL		BS_HATCHED, HS_VERTICAL						
#define BDIAGONAL2_FILL		(BYTE*)"\x44\x88\x11\x22\x44\x88\x11\x22"
#define CROSS2_FILL			(BYTE*)"\xff\x11\x11\x11\xff\x11\x11\x11"
#define DIAGCROSS2_FILL		(BYTE*)"\x55\x88\x55\x22\x55\x88\x55\x22"
#define DOT2_FILL			(BYTE*)"\x88\x00\x22\x00\x88\x00\x22\x00"
#define FDIAGONAL2_FILL		(BYTE*)"\x22\x11\x88\x44\x22\x11\x88\x44"
#define HORIZONTAL2_FILL	(BYTE*)"\x00\x00\xff\x00\x00\x00\xff\x00"
#define VERTICAL2_FILL		(BYTE*)"\x11\x11\x11\x11\x11\x11\x11\x11"
#define BDIAGONAL3_FILL		(BYTE*)"\xe0\xc1\x83\x07\x0e\x1c\x38\x70"
#define CROSS3_FILL			(BYTE*)"\x30\x30\x30\x30\x30\x30\xff\xff"
#define DIAGCROSS3_FILL		(BYTE*)"\xc7\x83\xc7\xee\x7c\x38\x7c\xee"
#define DOT3_FILL			(BYTE*)"\xc0\xc0\x0c\x0c\xc0\xc0\x0c\x0c"
#define FDIAGONAL3_FILL		(BYTE*)"\x07\x83\xc1\xe0\x70\x38\x1c\x0e"
#define HORIZONTAL3_FILL	(BYTE*)"\xff\xff\x00\x00\xff\xff\x00\x00"
#define VERTICAL3_FILL		(BYTE*)"\x33\x33\x33\x33\x33\x33\x33\x33"
#define INTERLEAVE_FILL		(BYTE*)"\xcc\x33\xcc\x33\xcc\x33\xcc\x33"



#if _MSC_VER > 1200 && _MSC_VER < 1900
#define _EASYX_DEPRECATE					__declspec(deprecated("This function is deprecated."))
#define _EASYX_DEPRECATE_WITHNEW(_NewFunc)	__declspec(deprecated("This function is deprecated. Instead, use this new function: " #_NewFunc ". See https:
#define _EASYX_DEPRECATE_OVERLOAD(_Func)	__declspec(deprecated("This overload is deprecated. See https:
#else
#define _EASYX_DEPRECATE
#define _EASYX_DEPRECATE_WITHNEW(_NewFunc)
#define _EASYX_DEPRECATE_OVERLOAD(_Func)
#endif

_EASYX_DEPRECATE_WITHNEW(settextstyle) void setfont(int nHeight, int nWidth, LPCTSTR lpszFace);
_EASYX_DEPRECATE_WITHNEW(settextstyle) void setfont(int nHeight, int nWidth, LPCTSTR lpszFace, int nEscapement, int nOrientation, int nWeight, bool bItalic, bool bUnderline, bool bStrikeOut);
_EASYX_DEPRECATE_WITHNEW(settextstyle) void setfont(int nHeight, int nWidth, LPCTSTR lpszFace, int nEscapement, int nOrientation, int nWeight, bool bItalic, bool bUnderline, bool bStrikeOut, BYTE fbCharSet, BYTE fbOutPrecision, BYTE fbClipPrecision, BYTE fbQuality, BYTE fbPitchAndFamily);
_EASYX_DEPRECATE_WITHNEW(settextstyle) void setfont(const LOGFONT *font);	
_EASYX_DEPRECATE_WITHNEW(gettextstyle) void getfont(LOGFONT *font);			

void bar(int left, int top, int right, int bottom);		
void bar3d(int left, int top, int right, int bottom, int depth, bool topflag);	

void drawpoly(int numpoints, const int *polypoints);	
void fillpoly(int numpoints, const int *polypoints);	

int getmaxx();					
int getmaxy();					

COLORREF getcolor();			
void setcolor(COLORREF color);	

void setwritemode(int mode);	

_EASYX_DEPRECATE	int	getx();								
_EASYX_DEPRECATE	int	gety();								
_EASYX_DEPRECATE	void moveto(int x, int y);				
_EASYX_DEPRECATE	void moverel(int dx, int dy);			
_EASYX_DEPRECATE	void lineto(int x, int y);				
_EASYX_DEPRECATE	void linerel(int dx, int dy);			
_EASYX_DEPRECATE	void outtext(LPCTSTR str);				
_EASYX_DEPRECATE	void outtext(TCHAR c);					

struct MOUSEMSG
{
UINT uMsg;				
bool mkCtrl		:1;		
bool mkShift	:1;		
bool mkLButton	:1;		
bool mkMButton	:1;		
bool mkRButton	:1;		
short x;				
short y;				
short wheel;			
};
_EASYX_DEPRECATE							bool MouseHit();			
_EASYX_DEPRECATE_WITHNEW(getmessage)		MOUSEMSG GetMouseMsg();		
_EASYX_DEPRECATE_WITHNEW(peekmessage)		bool PeekMouseMsg(MOUSEMSG *pMsg, bool bRemoveMsg = true);	
_EASYX_DEPRECATE_WITHNEW(flushmessage)		void FlushMouseMsgBuffer();	

typedef ExMessage EASYXMSG;	

#define EM_MOUSE	1
#define EM_KEY		2
#define EM_CHAR		4
#define EM_WINDOW	8