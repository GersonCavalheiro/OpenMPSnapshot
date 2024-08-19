#ifndef _DYNLINK_D3D11_H_
#define _DYNLINK_D3D11_H_
#include <windows.h>
#include <initguid.h>
#include <assert.h>
#include <wchar.h>
#include <mmsystem.h>
#include <commctrl.h> 
#include <shellapi.h> 
#include <new.h>      
#include <shlobj.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>
#if defined(DEBUG) || defined(_DEBUG)
#include <crtdbg.h>
#endif
#include <dxgi.h>
#include <d3d11.h>
#include <xinput.h>
#ifndef STRSAFE_NO_DEPRECATE
#pragma deprecated("strncpy")
#pragma deprecated("wcsncpy")
#pragma deprecated("_tcsncpy")
#pragma deprecated("wcsncat")
#pragma deprecated("strncat")
#pragma deprecated("_tcsncat")
#endif
#pragma warning( disable : 4996 ) 
#include <strsafe.h>
#pragma warning( default : 4996 )
typedef HRESULT(WINAPI *LPCREATEDXGIFACTORY)(REFIID, void **);
typedef HRESULT(WINAPI *LPD3D11CREATEDEVICEANDSWAPCHAIN)(__in_opt IDXGIAdapter *pAdapter, D3D_DRIVER_TYPE DriverType, HMODULE Software, UINT Flags, __in_ecount_opt(FeatureLevels) CONST D3D_FEATURE_LEVEL *pFeatureLevels, UINT FeatureLevels, UINT SDKVersion, __in_opt CONST DXGI_SWAP_CHAIN_DESC *pSwapChainDesc, __out_opt IDXGISwapChain **ppSwapChain, __out_opt ID3D11Device **ppDevice, __out_opt D3D_FEATURE_LEVEL *pFeatureLevel, __out_opt ID3D11DeviceContext **ppImmediateContext);
typedef HRESULT(WINAPI *LPD3D11CREATEDEVICE)(IDXGIAdapter *, D3D_DRIVER_TYPE, HMODULE, UINT32, D3D_FEATURE_LEVEL *, UINT, UINT32, ID3D11Device **, D3D_FEATURE_LEVEL *, ID3D11DeviceContext **);
static HMODULE                              s_hModDXGI = NULL;
static LPCREATEDXGIFACTORY                  sFnPtr_CreateDXGIFactory = NULL;
static HMODULE                              s_hModD3D11 = NULL;
static LPD3D11CREATEDEVICE                  sFnPtr_D3D11CreateDevice = NULL;
static LPD3D11CREATEDEVICEANDSWAPCHAIN      sFnPtr_D3D11CreateDeviceAndSwapChain = NULL;
static bool dynlinkUnloadD3D11API(void)
{
if (s_hModDXGI)
{
FreeLibrary(s_hModDXGI);
s_hModDXGI = NULL;
}
if (s_hModD3D11)
{
FreeLibrary(s_hModD3D11);
s_hModD3D11 = NULL;
}
return true;
}
static bool dynlinkLoadD3D11API(void)
{
if (s_hModD3D11 != NULL && s_hModDXGI != NULL)
{
return true;
}
#if 1
s_hModD3D11 = LoadLibrary("d3d11.dll");
if (s_hModD3D11 != NULL)
{
sFnPtr_D3D11CreateDevice = (LPD3D11CREATEDEVICE)GetProcAddress(s_hModD3D11, "D3D11CreateDevice");
sFnPtr_D3D11CreateDeviceAndSwapChain = (LPD3D11CREATEDEVICEANDSWAPCHAIN)GetProcAddress(s_hModD3D11, "D3D11CreateDeviceAndSwapChain");
}
else
{
printf("\nLoad d3d11.dll failed\n");
fflush(0);
}
if (!sFnPtr_CreateDXGIFactory)
{
s_hModDXGI = LoadLibrary("dxgi.dll");
if (s_hModDXGI)
{
sFnPtr_CreateDXGIFactory = (LPCREATEDXGIFACTORY)GetProcAddress(s_hModDXGI, "CreateDXGIFactory1");
}
return (s_hModDXGI != NULL) && (s_hModD3D11 != NULL);
}
return (s_hModD3D11 != NULL);
#else
sFnPtr_D3D11CreateDevice = (LPD3D11CREATEDEVICE)D3D11CreateDeviceAndSwapChain;
sFnPtr_D3D11CreateDeviceAndSwapChain = (LPD3D11CREATEDEVICEANDSWAPCHAIN)D3D11CreateDeviceAndSwapChain;
sFnPtr_D3DX11CompileFromMemory = (LPD3DX11COMPILEFROMMEMORY)D3DX11CompileFromMemory;
sFnPtr_CreateDXGIFactory = (LPCREATEDXGIFACTORY)CreateDXGIFactory;
return true;
#endif
return true;
}
#endif
