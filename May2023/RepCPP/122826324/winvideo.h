


#include <crtdbg.h> 

#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#ifndef _WIN32_WINNT
# define _WIN32_WINNT 0x0400
#endif
#if _WIN32_WINNT<0x0400
# define YIELD_TO_THREAD() Sleep(0)
#else
# define YIELD_TO_THREAD() SwitchToThread()
#endif
#include "video.h"
#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <fstream>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")

static const WORD MAX_CONSOLE_LINES = 500;
const COLORREF              RGBKEY = RGB(8, 8, 16); 
HWND                        g_hAppWnd;           
HANDLE                      g_handles[2] = {0,0};
unsigned int *              g_pImg = 0;          
int                         g_sizex, g_sizey;
static video *              g_video = 0;
WNDPROC                     g_pUserProc = 0;
HINSTANCE                   video::win_hInstance = 0;
int                         video::win_iCmdShow = 0;
static WNDCLASSEX *         gWndClass = 0;
static HACCEL               hAccelTable = 0;
static DWORD                g_msec = 0;
static int g_fps = 0, g_updates = 0, g_skips = 0;

bool DisplayError(LPSTR lpstrErr, HRESULT hres = 0); 
LRESULT CALLBACK InternalWndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

bool WinInit(HINSTANCE hInstance, int nCmdShow, WNDCLASSEX *uwc, const char *title, bool fixedsize)
{
WNDCLASSEX wndclass;  
if(uwc) {
memcpy(&wndclass, uwc, sizeof(wndclass));
g_pUserProc = uwc->lpfnWndProc;
} else {
memset(&wndclass, 0, sizeof(wndclass));
wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
wndclass.lpszClassName = title;
}
wndclass.cbSize = sizeof(wndclass);
wndclass.hInstance = hInstance;
wndclass.lpfnWndProc = InternalWndProc;
wndclass.style |= CS_HREDRAW | CS_VREDRAW;
wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);

_CrtSetDbgFlag(_CRTDBG_CHECK_ALWAYS_DF); 



if( !RegisterClassExA(&wndclass) ) return false;
int xaddend = GetSystemMetrics(fixedsize?SM_CXFIXEDFRAME:SM_CXFRAME)*2;
int yaddend = GetSystemMetrics(fixedsize?SM_CYFIXEDFRAME:SM_CYFRAME)*2 + GetSystemMetrics(SM_CYCAPTION);
if(wndclass.lpszMenuName) yaddend += GetSystemMetrics(SM_CYMENU);

g_hAppWnd = CreateWindowA(wndclass.lpszClassName,  
title,  
!fixedsize ? WS_OVERLAPPEDWINDOW :  
WS_OVERLAPPED|WS_CAPTION|WS_SYSMENU|WS_MINIMIZEBOX,
CW_USEDEFAULT,  
0,              
g_sizex+xaddend,
g_sizey+yaddend,
NULL,      
NULL,      
hInstance, 
NULL);     
return g_hAppWnd != NULL;
}

static bool RedirectIOToConsole(void)
{
int hConHandle; size_t lStdHandle;
CONSOLE_SCREEN_BUFFER_INFO coninfo;
FILE *fp;
AllocConsole();

GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &coninfo);
coninfo.dwSize.Y = MAX_CONSOLE_LINES;
SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE), coninfo.dwSize);

lStdHandle = (size_t)GetStdHandle(STD_OUTPUT_HANDLE);
hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
if(hConHandle <= 0) return false;
fp = _fdopen( hConHandle, "w" );
*stdout = *fp;
setvbuf( stdout, NULL, _IONBF, 0 );

lStdHandle = (size_t)GetStdHandle(STD_ERROR_HANDLE);
hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
if(hConHandle > 0) {
fp = _fdopen( hConHandle, "w" );
*stderr = *fp;
setvbuf( stderr, NULL, _IONBF, 0 );
}

lStdHandle = (size_t)GetStdHandle(STD_INPUT_HANDLE);
hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
if(hConHandle > 0) {
fp = _fdopen( hConHandle, "r" );
*stdin = *fp;
setvbuf( stdin, NULL, _IONBF, 0 );
}

std::ios::sync_with_stdio();
return true;
}


video::video()
: red_mask(0xff0000), red_shift(16), green_mask(0xff00),
green_shift(8), blue_mask(0xff), blue_shift(0), depth(24)
{
assert(g_video == 0);
g_video = this; title = "Video"; running = threaded = calc_fps = false; updating = true;
}

void video::win_set_class(WNDCLASSEX &wcex)
{
gWndClass = &wcex;
}

void video::win_load_accelerators(int idc)
{
hAccelTable = LoadAccelerators(win_hInstance, MAKEINTRESOURCE(idc));
}

bool video::init_console()
{
if(RedirectIOToConsole()) {
if(!g_pImg && g_sizex && g_sizey)
g_pImg = new unsigned int[g_sizex * g_sizey];
if(g_pImg) running = true;
return true;
}
return false;
}

video::~video()
{
if(g_video) terminate();
}

DWORD WINAPI thread_video(LPVOID lpParameter)
{
video *v = (video*)lpParameter;
v->on_process();
return 0;
}

static bool loop_once(video *v)
{
if(int updates = g_updates) {
g_updates = 0;
if(g_video->updating) { g_skips += updates-1; g_fps++; }
else g_skips += updates;
UpdateWindow(g_hAppWnd);
}
DWORD msec = GetTickCount();
if(v->calc_fps && msec >= g_msec+1000) {
double sec = (msec - g_msec)/1000.0;
char buffer[256], n = _snprintf(buffer, 128, "%s: %d fps", v->title, int(double(g_fps + g_skips)/sec));
if(g_skips) _snprintf(buffer+n, 128, " - %d skipped = %d updates", int(g_skips/sec), int(g_fps/sec));
SetWindowTextA(g_hAppWnd, buffer);
g_msec = msec; g_skips = g_fps = 0;
}
MSG msg;
if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
{
if( msg.message == WM_QUIT ) { v->running = false; return false; }
if( !hAccelTable || !TranslateAccelerator(msg.hwnd, hAccelTable, &msg) )
{
TranslateMessage(&msg);
DispatchMessage(&msg);
}
return true; 
}
return false;
}

void video::main_loop()
{
InvalidateRect(g_hAppWnd, 0, false);
g_msec = GetTickCount(); 
while(g_msec + 500 > GetTickCount()) { loop_once(this); Sleep(1); }
g_msec = GetTickCount();
if(threaded) {
g_handles[0] = CreateThread (
NULL,             
0,                
(LPTHREAD_START_ROUTINE) thread_video,
this,               
0, 0);
if(!g_handles[0]) { DisplayError("Can't create thread"); return; }
else 
g_handles[1] = CreateEvent(NULL, false, false, NULL);
while(running) {
while(loop_once(this));
YIELD_TO_THREAD(); 
DWORD r = MsgWaitForMultipleObjects(2, g_handles, false, INFINITE, QS_ALLINPUT^QS_MOUSEMOVE);
if(r == WAIT_OBJECT_0) break; 
}
running = false;
if(WaitForSingleObject(g_handles[0], 300) == WAIT_TIMEOUT)
TerminateThread(g_handles[0], 0);
if(g_handles[0]) CloseHandle(g_handles[0]);
if(g_handles[1]) CloseHandle(g_handles[1]);
g_handles[0] = g_handles[1] = 0;
}
else on_process();
}

bool video::next_frame()
{
if(!running) return false;
if(!threaded) while(loop_once(this));
else if(g_handles[1]) {
SetEvent(g_handles[1]);
YIELD_TO_THREAD();
}
return true;
}

void video::show_title()
{
if(g_hAppWnd)
SetWindowTextA(g_hAppWnd, title);
}
