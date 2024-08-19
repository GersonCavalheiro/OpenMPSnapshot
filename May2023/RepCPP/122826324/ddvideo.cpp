

#include "winvideo.h"
#include <cassert>
#include <stdio.h>
#include <ddraw.h>

#pragma comment(lib, "ddraw.lib")
#pragma comment(lib, "dxguid.lib")

LPDIRECTDRAW7               g_pDD = NULL;        
LPDIRECTDRAWSURFACE7        g_pDDSPrimary = NULL;
LPDIRECTDRAWSURFACE7        g_pDDSBack = NULL;   
LPDIRECTDRAWSURFACE7        g_pDDSOverlay = NULL;
LPDIRECTDRAWCLIPPER         g_pClipper = NULL;   
DDOVERLAYFX                 g_OverlayFX;         
DDCAPS                      g_DDCaps;            
DWORD                       g_OverlayFlags = 0;  
DWORD                       g_dwXRatio,
g_dwYRatio;          
RECT                        g_rcSrc = {0, 0, 0, 0},
g_rcDst = {0, 0, 0, 0};
HANDLE                      g_hVSync;

#ifdef DDSCAPS_PRIMARYSURFACELEFT
#include <dxerr8.h>
#pragma comment(lib, "dxerr8.lib")
#else
#include <d3dx.h>
#pragma comment(lib, "d3dx.lib")
#endif

bool DisplayError(LPSTR lpstrErr, HRESULT hres)
{
static bool InError = false;
int retval = 0;
if (!InError)
{
InError = true;
#ifdef DDSCAPS_PRIMARYSURFACELEFT
const char *message = hres?DXGetErrorString8A(hres):0;
#else
char message[256]; if(hres) D3DXGetErrorString(hres, 256, message);
#endif
retval = MessageBoxA(g_hAppWnd, lpstrErr, hres?message:"Error!", MB_OK|MB_ICONERROR);
InError = false;
}
return false;
}

void DestroyOverlay()
{
if (g_pClipper)
g_pClipper->Release();
if (g_pDDSOverlay) {
g_pImg = 0; LPDIRECTDRAWSURFACE7 pDDSOverlay(g_pDDSOverlay);
g_pDDSOverlay = NULL;
YIELD_TO_THREAD();
pDDSOverlay->Release(); 
}
}

void DestroyPrimary()
{
if (g_pDDSPrimary)
{
g_pDDSPrimary->Release();
g_pDDSPrimary = NULL;
}
}

void DestroyDDraw()
{
DestroyPrimary();
if (g_pDD) {
LPDIRECTDRAW7 pDD(g_pDD); 
g_pDD = NULL; Sleep(1); pDD->Release();
}
}

void CheckBoundries(void)
{
if ((g_DDCaps.dwCaps & DDCAPS_OVERLAYSTRETCH) && (g_DDCaps.dwMinOverlayStretch)
&& (g_dwXRatio < g_DDCaps.dwMinOverlayStretch))
{
g_rcDst.right = 2 * GetSystemMetrics(SM_CXSIZEFRAME) + g_rcDst.left + (g_sizex
* (g_DDCaps.dwMinOverlayStretch + 1)) / 1000;
SetWindowTextA(g_hAppWnd, "Window is too small!");
}
else if ((g_DDCaps.dwCaps & DDCAPS_OVERLAYSTRETCH) && (g_DDCaps.dwMaxOverlayStretch)
&& (g_dwXRatio > g_DDCaps.dwMaxOverlayStretch))
{
g_rcDst.right = 2 * GetSystemMetrics(SM_CXSIZEFRAME) + g_rcDst.left + (g_sizey
* (g_DDCaps.dwMaxOverlayStretch + 999)) / 1000;
SetWindowTextA(g_hAppWnd, "Window is too large!");
}
else if(!g_video->calc_fps) SetWindowText(g_hAppWnd, g_video->title);

g_dwXRatio = (g_rcDst.right - g_rcDst.left) * 1000 / (g_rcSrc.right - g_rcSrc.left);
g_dwYRatio = (g_rcDst.bottom - g_rcDst.top) * 1000 / (g_rcSrc.bottom - g_rcSrc.top);

if (g_rcDst.left < 0)
{
g_rcSrc.left = -g_rcDst.left * 1000 / g_dwXRatio;
g_rcDst.left = 0;
}
if (g_rcDst.right > GetSystemMetrics(SM_CXSCREEN))
{
g_rcSrc.right = g_sizex - ((g_rcDst.right - GetSystemMetrics(SM_CXSCREEN)) * 1000 / g_dwXRatio);
g_rcDst.right = GetSystemMetrics(SM_CXSCREEN);
}
if (g_rcDst.bottom > GetSystemMetrics(SM_CYSCREEN))
{
g_rcSrc.bottom = g_sizey - ((g_rcDst.bottom - GetSystemMetrics(SM_CYSCREEN)) * 1000 / g_dwYRatio);
g_rcDst.bottom = GetSystemMetrics(SM_CYSCREEN);
}
if (g_rcDst.top < 0)
{
g_rcSrc.top = -g_rcDst.top * 1000 / g_dwYRatio;
g_rcDst.top = 0;
}

if ((g_DDCaps.dwCaps & DDCAPS_ALIGNBOUNDARYSRC) && g_DDCaps.dwAlignBoundarySrc)
g_rcSrc.left = (g_rcSrc.left + g_DDCaps.dwAlignBoundarySrc / 2) & -(signed)
(g_DDCaps.dwAlignBoundarySrc);
if ((g_DDCaps.dwCaps & DDCAPS_ALIGNSIZESRC) && g_DDCaps.dwAlignSizeSrc)
g_rcSrc.right = g_rcSrc.left + (g_rcSrc.right - g_rcSrc.left + g_DDCaps.dwAlignSizeSrc
/ 2) & -(signed) (g_DDCaps.dwAlignSizeSrc);
if ((g_DDCaps.dwCaps & DDCAPS_ALIGNBOUNDARYDEST) && g_DDCaps.dwAlignBoundaryDest)
g_rcDst.left = (g_rcDst.left + g_DDCaps.dwAlignBoundaryDest / 2) & -(signed)
(g_DDCaps.dwAlignBoundaryDest);
if ((g_DDCaps.dwCaps & DDCAPS_ALIGNSIZEDEST) && g_DDCaps.dwAlignSizeDest)
g_rcDst.right = g_rcDst.left + (g_rcDst.right - g_rcDst.left) & -(signed) (g_DDCaps.dwAlignSizeDest);
}

DWORD DDColorMatch(IDirectDrawSurface7 * pdds, COLORREF rgb)
{
COLORREF       rgbT;
HDC            hdc;
DWORD          dw = CLR_INVALID;
DDSURFACEDESC2 ddsd;
HRESULT        hres;

if (rgb != CLR_INVALID && pdds->GetDC(&hdc) == DD_OK) {
rgbT = GetPixel(hdc, 0, 0);     
SetPixel(hdc, 0, 0, rgb);       
pdds->ReleaseDC(hdc);
}
ddsd.dwSize = sizeof(ddsd);
while ((hres = pdds->Lock(NULL, &ddsd, 0, NULL)) == DDERR_WASSTILLDRAWING)
YIELD_TO_THREAD();
if (hres == DD_OK) {
dw = *(DWORD *) ddsd.lpSurface;                 
if (ddsd.ddpfPixelFormat.dwRGBBitCount < 32)
dw &= (1 << ddsd.ddpfPixelFormat.dwRGBBitCount) - 1;  
pdds->Unlock(NULL);
}
else return DisplayError("Can't lock primary surface", hres);
if (rgb != CLR_INVALID && pdds->GetDC(&hdc) == DD_OK) {
SetPixel(hdc, 0, 0, rgbT);
pdds->ReleaseDC(hdc);
}
return dw;
}

bool DrawOverlay()
{
HRESULT        hRet;       
DDSURFACEDESC2 surfDesc;
memset(&surfDesc, 0, sizeof(surfDesc)); surfDesc.dwSize = sizeof(surfDesc);

hRet = g_pDDSOverlay->Lock(NULL, &surfDesc, DDLOCK_SURFACEMEMORYPTR | DDLOCK_NOSYSLOCK | DDLOCK_WRITEONLY, NULL);
if (hRet != DD_OK ||  surfDesc.lpSurface == NULL)
return DisplayError("Can't lock overlay surface", hRet);
else {
g_pImg = (unsigned int *)surfDesc.lpSurface;
}
memset(&g_OverlayFX, 0, sizeof(g_OverlayFX)); g_OverlayFX.dwSize = sizeof(g_OverlayFX);
g_OverlayFlags = DDOVER_SHOW;
if ((g_DDCaps.dwCKeyCaps & DDCKEYCAPS_DESTOVERLAY) && ((g_DDCaps.dwCaps & DDCAPS_OVERLAYCANTCLIP) || (g_DDCaps.dwCKeyCaps & DDCKEYCAPS_NOCOSTOVERLAY) ))
{
g_OverlayFX.dckDestColorkey.dwColorSpaceLowValue =
g_OverlayFX.dckDestColorkey.dwColorSpaceHighValue = DDColorMatch(g_pDDSPrimary, RGBKEY);
g_OverlayFlags |= DDOVER_DDFX | DDOVER_KEYDESTOVERRIDE;
} else {
hRet = g_pDD->CreateClipper(0, &g_pClipper, NULL);
if (hRet != DD_OK)
return DisplayError("Can't create clipper", hRet);
hRet = g_pClipper->SetHWnd(0, g_hAppWnd);
if (hRet != DD_OK)
return DisplayError("Can't attach clipper", hRet);
hRet = g_pDDSPrimary->SetClipper(g_pClipper);
if (hRet != DD_OK)
return DisplayError("Can't set clipper", hRet);
}
return true;
}

bool DDPrimaryInit()
{
HRESULT        hRet;
DDSURFACEDESC2 ddsd;  

memset(&ddsd, 0, sizeof(ddsd)); 
ddsd.dwSize = sizeof(ddsd);     
ddsd.dwFlags = DDSD_CAPS;       
ddsd.ddsCaps.dwCaps = DDSCAPS_PRIMARYSURFACE;  
hRet = g_pDD->CreateSurface(&ddsd, &g_pDDSPrimary, NULL);
if (hRet != DD_OK)
return DisplayError("Can't create primary surface", hRet);
return true;
}

bool DDInit()
{
HRESULT hRet;
g_rcSrc.right = g_sizex;
g_rcSrc.bottom = g_sizey;

hRet = DirectDrawCreateEx(NULL, (VOID**)&g_pDD, IID_IDirectDraw7, NULL);
if (hRet != DD_OK)
return DisplayError("Can't create DirectDraw7 instance", hRet);

hRet = g_pDD->SetCooperativeLevel(g_hAppWnd, DDSCL_NORMAL);
if (hRet != DD_OK)
return DisplayError("Can't set cooperative level", hRet);
return DDPrimaryInit();
}

bool DDOverlayInit()
{
memset(&g_DDCaps, 0, sizeof(g_DDCaps));
g_DDCaps.dwSize = sizeof(g_DDCaps);
if (g_pDD->GetCaps(&g_DDCaps, 0))
return DisplayError("Can't get capabilities");

if (!(g_DDCaps.dwCaps & DDCAPS_OVERLAY))
return DisplayError("Hardware doesn't support overlays");


DDSURFACEDESC2              ddsd;  
HRESULT                     hRet;  
DDPIXELFORMAT               ddpfOverlayFormats[] = {
{sizeof(DDPIXELFORMAT), DDPF_RGB, 0, 32, 0xFF0000, 0x0FF00, 0x0000FF, 0}, 
{sizeof(DDPIXELFORMAT), DDPF_RGB, 0, 16, 0x007C00, 0x003e0, 0x00001F, 0}, 
{sizeof(DDPIXELFORMAT), DDPF_RGB, 0, 16, 0x00F800, 0x007e0, 0x00001F, 0}, 
{sizeof(DDPIXELFORMAT), DDPF_FOURCC, mmioFOURCC('U','Y','V','Y'), 16, 0, 0, 0, 0}, 
{sizeof(DDPIXELFORMAT), DDPF_FOURCC, mmioFOURCC('Y','4','2','2'), 16, 0, 0, 0, 0}, 
{sizeof(DDPIXELFORMAT), DDPF_FOURCC, mmioFOURCC('Y','U','Y','2'), 16, 0, 0, 0, 0}, 
{0}};

memset(&ddsd, 0, sizeof(ddsd));
ddsd.dwSize = sizeof(ddsd);
ddsd.ddsCaps.dwCaps = DDSCAPS_OVERLAY | g_DDCaps.ddsCaps.dwCaps&DDSCAPS_VIDEOMEMORY;
ddsd.dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT;
ddsd.dwBackBufferCount = 0;
ddsd.dwWidth = g_sizex;
ddsd.dwHeight = g_sizey;
for(int format = 0; ddpfOverlayFormats[format].dwSize; format++) {
ddsd.ddpfPixelFormat = ddpfOverlayFormats[format];
hRet = g_pDD->CreateSurface(&ddsd, &g_pDDSOverlay, NULL);
if(hRet == DD_OK) break;
}
if (hRet != DD_OK)
return DisplayError("Can't create appropriate overlay surface", hRet);
return true;
}

inline void mouse(int k, LPARAM lParam)
{
int x = (int)LOWORD(lParam), y = (int)HIWORD(lParam);
g_video->on_mouse( x*g_sizex/(g_rcDst.right - g_rcDst.left),
y*g_sizey/(g_rcDst.bottom - g_rcDst.top), k);
}

LRESULT CALLBACK InternalWndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
PAINTSTRUCT                 ps;         
POINT                       p = {0, 0}; 
HRESULT                     hRet;

switch (iMsg)
{
case WM_MOVE:
if (!IsIconic(hwnd))
{
g_rcSrc.left = 0;
g_rcSrc.right = g_sizex;
g_rcSrc.top = 0;
g_rcSrc.bottom = g_sizey;
GetClientRect(hwnd, &g_rcDst);
g_dwXRatio = (g_rcDst.right - g_rcDst.left) * 1000 /
(g_rcSrc.right - g_rcSrc.left);
g_dwYRatio = (g_rcDst.bottom - g_rcDst.top) * 1000 /
(g_rcSrc.bottom - g_rcSrc.top);
ClientToScreen(hwnd, &p);
g_rcDst.left = p.x;
g_rcDst.top = p.y;
g_rcDst.bottom += p.y;
g_rcDst.right += p.x;
CheckBoundries();
}
else
if (g_pDDSOverlay && g_pDDSPrimary)
g_pDDSOverlay->UpdateOverlay(NULL, g_pDDSPrimary, NULL, DDOVER_HIDE, NULL);
if (hwnd)
{
InvalidateRect(hwnd, NULL, FALSE);
UpdateWindow(hwnd);
}
return 0L;

case WM_SIZE:
if (wParam != SIZE_MINIMIZED)
{
GetClientRect(hwnd, &g_rcDst);
ClientToScreen(hwnd, &p);
g_rcDst.left = p.x;
g_rcDst.top = p.y;
g_rcDst.bottom += p.y;
g_rcDst.right += p.x;
g_rcSrc.left = 0;
g_rcSrc.right = g_sizex;
g_rcSrc.top = 0;
g_rcSrc.bottom = g_sizey;
g_dwXRatio = (g_rcDst.right - g_rcDst.left) * 1000 /
(g_rcSrc.right - g_rcSrc.left);
g_dwYRatio = (g_rcDst.bottom - g_rcDst.top) * 1000 /
(g_rcSrc.bottom - g_rcSrc.top);
CheckBoundries();
}
return 0L;

case WM_PAINT:
BeginPaint(hwnd, &ps);
if (!g_pDDSPrimary || (g_pDDSPrimary->IsLost() != DD_OK) ||
(g_pDDSOverlay == NULL))
{
DestroyOverlay();
DestroyPrimary();
if (DDPrimaryInit())
if (DDOverlayInit())
if (!DrawOverlay())
DestroyOverlay();
}
if (g_pDDSOverlay && g_pDDSPrimary && g_video->updating)
{
hRet = g_pDDSOverlay->UpdateOverlay(&g_rcSrc, g_pDDSPrimary,
&g_rcDst, g_OverlayFlags,
&g_OverlayFX);
#ifdef _DEBUG
if(hRet != DD_OK) DisplayError("Can't update overlay", hRet);
#endif
}
EndPaint(hwnd, &ps);
return 0L;

case WM_LBUTTONDOWN:    mouse(1, lParam); break;
case WM_LBUTTONUP:      mouse(-1, lParam); break;
case WM_RBUTTONDOWN:    mouse(2, lParam); break;
case WM_RBUTTONUP:      mouse(-2, lParam); break;
case WM_MBUTTONDOWN:    mouse(3, lParam); break;
case WM_MBUTTONUP:      mouse(-3, lParam); break;
case WM_CHAR:           g_video->on_key(wParam); break;

case WM_DISPLAYCHANGE:  return 0L;

case WM_DESTROY:
PostQuitMessage(0);
return 0L;
}
return g_pUserProc? g_pUserProc(hwnd, iMsg, wParam, lParam) : DefWindowProc(hwnd, iMsg, wParam, lParam);
}

DWORD WINAPI thread_vsync(LPVOID lpParameter)
{
BOOL vblank = false;
while(g_video && g_video->running) {
while(!vblank && g_video && g_video->running) {
YIELD_TO_THREAD();
LPDIRECTDRAW7 pDD(g_pDD);
if(pDD) pDD->GetVerticalBlankStatus(&vblank);
}
LPDIRECTDRAWSURFACE7 pDDSOverlay(g_pDDSOverlay);
if(pDDSOverlay) pDDSOverlay->UpdateOverlay(&g_rcSrc, g_pDDSPrimary, &g_rcDst, g_OverlayFlags | DDOVER_REFRESHALL, &g_OverlayFX);
do {
Sleep(1);
LPDIRECTDRAW7 pDD(g_pDD);
if(pDD) pDD->GetVerticalBlankStatus(&vblank);
} while(vblank && g_video && g_video->running);
while(g_video && !g_video->updating && g_video->running) Sleep(10);
}
return 0;
}


inline void mask2bits(unsigned int mask, color_t &save, char &shift)
{
save  = mask; if(!mask) { shift = 8; return; }
shift = 0; while(!(mask&1)) ++shift, mask >>= 1;
int bits = 0; while(mask&1) ++bits,  mask >>= 1;
shift += bits - 8;
}

bool video::init_window(int sizex, int sizey)
{
assert(win_hInstance != 0);
g_sizex = sizex; g_sizey = sizey;
if( !WinInit(win_hInstance, win_iCmdShow, gWndClass, title, false) )
return DisplayError("Unable to initialize the program's window.");
running = true;
if( !DDInit() ) {
DestroyDDraw();
goto fail;
}
if( !DDOverlayInit() || !DrawOverlay() ) {
DestroyOverlay();
DestroyDDraw();
goto fail;
}
DDPIXELFORMAT PixelFormat; memset(&PixelFormat, 0, sizeof(PixelFormat)); PixelFormat.dwSize = sizeof(PixelFormat);
g_pDDSOverlay->GetPixelFormat(&PixelFormat);
mask2bits(PixelFormat.dwRBitMask, red_mask, red_shift);
mask2bits(PixelFormat.dwGBitMask, green_mask, green_shift);
mask2bits(PixelFormat.dwBBitMask, blue_mask, blue_shift);
if(PixelFormat.dwFlags == DDPF_RGB)
depth = char(PixelFormat.dwRGBBitCount);
else depth = -char(PixelFormat.dwFourCC);
for(int i = 0, e = sizex * sizey * PixelFormat.dwRGBBitCount / 32, c = get_color(0, 0, 0); i < e; i++)
g_pImg[i] = c; 
ShowWindow(g_hAppWnd, SW_SHOW);
g_hVSync = CreateThread (
NULL,          
0,             
(LPTHREAD_START_ROUTINE) thread_vsync,
this,               
0, 0);
SetPriorityClass(g_hVSync, IDLE_PRIORITY_CLASS); 
return true;
fail:
g_pImg = new unsigned int[g_sizex * g_sizey];
return false;
}

void video::terminate()
{
running = false;
DestroyOverlay();
if(WaitForSingleObject(g_hVSync, 100) == WAIT_TIMEOUT) TerminateThread(g_hVSync, 0);
CloseHandle(g_hVSync);
DestroyDDraw();
if(g_pImg) delete[] g_pImg;
g_pImg = 0; g_video = 0;
}

drawing_area::drawing_area(int x, int y, int sizex, int sizey)
: start_x(x), start_y(y), size_x(sizex), size_y(sizey), pixel_depth(g_video->depth),
base_index(y*g_sizex + x), max_index(g_sizex*g_sizey), index_stride(g_sizex), ptr32(g_pImg)
{
assert(ptr32); assert(x < g_sizex); assert(y < g_sizey);
assert(x+sizex <= g_sizex); assert(y+sizey <= g_sizey);

index = base_index; 
}

drawing_area::~drawing_area()
{
}
