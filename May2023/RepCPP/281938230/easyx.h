

#pragma once

#ifndef WINVER
#define WINVER 0x0400			
#endif

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0500		
#endif

#ifndef _WIN32_WINDOWS
#define _WIN32_WINDOWS 0x0410	
#endif

#ifdef UNICODE
#pragma comment(lib,"EasyXw.lib")
#else
#pragma comment(lib,"EasyXa.lib")
#endif


#ifndef __cplusplus
#error EasyX is only for C++
#endif

#include <windows.h>
#include <tchar.h>

#define EX_SHOWCONSOLE		1		
#define EX_NOCLOSE			2		
#define EX_NOMINIMIZE		4		
#define EX_DBLCLKS			8		


#define	BLACK			0
#define	BLUE			0xAA0000
#define	GREEN			0x00AA00
#define	CYAN			0xAAAA00
#define	RED				0x0000AA
#define	MAGENTA			0xAA00AA
#define	BROWN			0x0055AA
#define	LIGHTGRAY		0xAAAAAA
#define	DARKGRAY		0x555555
#define	LIGHTBLUE		0xFF5555
#define	LIGHTGREEN		0x55FF55
#define	LIGHTCYAN		0xFFFF55
#define	LIGHTRED		0x5555FF
#define	LIGHTMAGENTA	0xFF55FF
#define	YELLOW			0x55FFFF
#define	WHITE			0xFFFFFF

#define BGR(color)	( (((color) & 0xFF) << 16) | ((color) & 0xFF00FF00) | (((color) & 0xFF0000) >> 16) )


class IMAGE;

class LINESTYLE
{
public:
LINESTYLE();
LINESTYLE(const LINESTYLE &style);
LINESTYLE& operator = (const LINESTYLE &style);
virtual ~LINESTYLE();

DWORD	style;
DWORD	thickness;
DWORD	*puserstyle;
DWORD	userstylecount;
};

class FILLSTYLE
{
public:
FILLSTYLE();
FILLSTYLE(const FILLSTYLE &style);
FILLSTYLE& operator = (const FILLSTYLE &style);
virtual ~FILLSTYLE();

int			style;				
long		hatch;				
IMAGE*		ppattern;			
};

class IMAGE
{
public:
int getwidth() const;			
int getheight() const;			

private:
int			width, height;		
HBITMAP		m_hBmp;
HDC			m_hMemDC;
float		m_data[6];
COLORREF	m_LineColor;		
COLORREF	m_FillColor;		
COLORREF	m_TextColor;		
COLORREF	m_BkColor;			
DWORD*		m_pBuffer;			

LINESTYLE	m_LineStyle;		
FILLSTYLE	m_FillStyle;		

virtual void SetDefault();		

public:
IMAGE(int _width = 0, int _height = 0);
IMAGE(const IMAGE &img);
IMAGE& operator = (const IMAGE &img);
virtual ~IMAGE();
virtual void Resize(int _width, int _height);			
};




HWND initgraph(int width, int height, int flag = 0);		
void closegraph();											



void cleardevice();											
void setcliprgn(HRGN hrgn);									
void clearcliprgn();										

void getlinestyle(LINESTYLE* pstyle);						
void setlinestyle(const LINESTYLE* pstyle);					
void setlinestyle(int style, int thickness = 1, const DWORD *puserstyle = NULL, DWORD userstylecount = 0);	
void getfillstyle(FILLSTYLE* pstyle);						
void setfillstyle(const FILLSTYLE* pstyle);					
void setfillstyle(int style, long hatch = NULL, IMAGE* ppattern = NULL);		
void setfillstyle(BYTE* ppattern8x8);						

void setorigin(int x, int y);								
void getaspectratio(float *pxasp, float *pyasp);			
void setaspectratio(float xasp, float yasp);				

int  getrop2();						
void setrop2(int mode);				
int  getpolyfillmode();				
void setpolyfillmode(int mode);		

void graphdefaults();				

COLORREF getlinecolor();			
void setlinecolor(COLORREF color);	
COLORREF gettextcolor();			
void settextcolor(COLORREF color);	
COLORREF getfillcolor();			
void setfillcolor(COLORREF color);	
COLORREF getbkcolor();				
void setbkcolor(COLORREF color);	
int  getbkmode();					
void setbkmode(int mode);			

COLORREF RGBtoGRAY(COLORREF rgb);
void RGBtoHSL(COLORREF rgb, float *H, float *S, float *L);
void RGBtoHSV(COLORREF rgb, float *H, float *S, float *V);
COLORREF HSLtoRGB(float H, float S, float L);
COLORREF HSVtoRGB(float H, float S, float V);



COLORREF getpixel(int x, int y);				
void putpixel(int x, int y, COLORREF color);	

void line(int x1, int y1, int x2, int y2);		

void rectangle	   (int left, int top, int right, int bottom);	
void fillrectangle (int left, int top, int right, int bottom);	
void solidrectangle(int left, int top, int right, int bottom);	
void clearrectangle(int left, int top, int right, int bottom);	

void circle		(int x, int y, int radius);		
void fillcircle (int x, int y, int radius);		
void solidcircle(int x, int y, int radius);		
void clearcircle(int x, int y, int radius);		

void ellipse	 (int left, int top, int right, int bottom);	
void fillellipse (int left, int top, int right, int bottom);	
void solidellipse(int left, int top, int right, int bottom);	
void clearellipse(int left, int top, int right, int bottom);	

void roundrect	   (int left, int top, int right, int bottom, int ellipsewidth, int ellipseheight);		
void fillroundrect (int left, int top, int right, int bottom, int ellipsewidth, int ellipseheight);		
void solidroundrect(int left, int top, int right, int bottom, int ellipsewidth, int ellipseheight);		
void clearroundrect(int left, int top, int right, int bottom, int ellipsewidth, int ellipseheight);		

void arc	 (int left, int top, int right, int bottom, double stangle, double endangle);	
void pie	 (int left, int top, int right, int bottom, double stangle, double endangle);	
void fillpie (int left, int top, int right, int bottom, double stangle, double endangle);	
void solidpie(int left, int top, int right, int bottom, double stangle, double endangle);	
void clearpie(int left, int top, int right, int bottom, double stangle, double endangle);	

void polyline	 (const POINT *points, int num);								
void polygon	 (const POINT *points, int num);								
void fillpolygon (const POINT *points, int num);								
void solidpolygon(const POINT *points, int num);								
void clearpolygon(const POINT *points, int num);								

void polybezier(const POINT *points, int num);									
void floodfill(int x, int y, COLORREF color, int filltype = FLOODFILLBORDER);	



void outtextxy(int x, int y, LPCTSTR str);				
void outtextxy(int x, int y, TCHAR c);					
int textwidth(LPCTSTR str);								
int textwidth(TCHAR c);									
int textheight(LPCTSTR str);							
int textheight(TCHAR c);								
int drawtext(LPCTSTR str, RECT* pRect, UINT uFormat);	
int drawtext(TCHAR c, RECT* pRect, UINT uFormat);		

void settextstyle(int nHeight, int nWidth, LPCTSTR lpszFace);
void settextstyle(int nHeight, int nWidth, LPCTSTR lpszFace, int nEscapement, int nOrientation, int nWeight, bool bItalic, bool bUnderline, bool bStrikeOut);
void settextstyle(int nHeight, int nWidth, LPCTSTR lpszFace, int nEscapement, int nOrientation, int nWeight, bool bItalic, bool bUnderline, bool bStrikeOut, BYTE fbCharSet, BYTE fbOutPrecision, BYTE fbClipPrecision, BYTE fbQuality, BYTE fbPitchAndFamily);
void settextstyle(const LOGFONT *font);	
void gettextstyle(LOGFONT *font);		



void loadimage(IMAGE *pDstImg, LPCTSTR pImgFile, int nWidth = 0, int nHeight = 0, bool bResize = false);					
void loadimage(IMAGE *pDstImg, LPCTSTR pResType, LPCTSTR pResName, int nWidth = 0, int nHeight = 0, bool bResize = false);	
void saveimage(LPCTSTR pImgFile, IMAGE* pImg = NULL);																		
void getimage(IMAGE *pDstImg, int srcX, int srcY, int srcWidth, int srcHeight);												
void putimage(int dstX, int dstY, const IMAGE *pSrcImg, DWORD dwRop = SRCCOPY);												
void putimage(int dstX, int dstY, int dstWidth, int dstHeight, const IMAGE *pSrcImg, int srcX, int srcY, DWORD dwRop = SRCCOPY);		
void rotateimage(IMAGE *dstimg, IMAGE *srcimg, double radian, COLORREF bkcolor = BLACK, bool autosize = false, bool highquality = true);
void Resize(IMAGE* pImg, int width, int height);	
DWORD* GetImageBuffer(IMAGE* pImg = NULL);			
IMAGE* GetWorkingImage();							
void SetWorkingImage(IMAGE* pImg = NULL);			
HDC GetImageHDC(IMAGE* pImg = NULL);				



int	getwidth();			
int	getheight();		

void BeginBatchDraw();	
void FlushBatchDraw();	
void FlushBatchDraw(int left, int top, int right, int bottom);	
void EndBatchDraw();	
void EndBatchDraw(int left, int top, int right, int bottom);	

HWND GetHWnd();								
const TCHAR* GetEasyXVer();						

bool InputBox(LPTSTR pString, int nMaxCount, LPCTSTR pPrompt = NULL, LPCTSTR pTitle = NULL, LPCTSTR pDefault = NULL, int width = 0, int height = 0, bool bOnlyOK = true);




#define EX_MOUSE	1
#define EX_KEY		2
#define EX_CHAR		4
#define EX_WINDOW	8

struct ExMessage
{
USHORT message;					
union
{
struct
{
bool ctrl		:1;		
bool shift		:1;		
bool lbutton	:1;		
bool mbutton	:1;		
bool rbutton	:1;		
short x;				
short y;				
short wheel;			
};

struct
{
BYTE vkcode;			
BYTE scancode;			
bool extended	:1;		
bool prevdown	:1;		
};

TCHAR ch;

struct
{
WPARAM wParam;
LPARAM lParam;
};
};
};

ExMessage getmessage(BYTE filter = -1);										
void getmessage(ExMessage *msg, BYTE filter = -1);							
bool peekmessage(ExMessage *msg, BYTE filter = -1, bool removemsg = true);	
void flushmessage(BYTE filter = -1);										
