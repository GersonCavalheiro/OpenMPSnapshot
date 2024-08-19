#ifndef _DYNLINK_D3D10_H_
#define _DYNLINK_D3D10_H_
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
#include <d3d9.h>
#include <dxgi.h>
#include <d3d10_1.h>
#include <d3d10.h>
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
#include <DirectXMath.h>
using namespace DirectX;
struct DXUTD3D9DeviceSettings
{
UINT AdapterOrdinal;
D3DDEVTYPE DeviceType;
D3DFORMAT AdapterFormat;
DWORD BehaviorFlags;
D3DPRESENT_PARAMETERS pp;
};
struct DXUTD3D10DeviceSettings
{
UINT AdapterOrdinal;
D3D10_DRIVER_TYPE DriverType;
UINT Output;
DXGI_SWAP_CHAIN_DESC sd;
UINT32 CreateFlags;
UINT32 SyncInterval;
DWORD PresentFlags;
bool AutoCreateDepthStencil; 
DXGI_FORMAT AutoDepthStencilFormat;
};
enum DXUTDeviceVersion { DXUT_D3D9_DEVICE, DXUT_D3D10_DEVICE };
struct DXUTDeviceSettings
{
DXUTDeviceVersion ver;
union
{
DXUTD3D9DeviceSettings d3d9; 
DXUTD3D10DeviceSettings d3d10; 
};
};
#define DXUTERR_NODIRECT3D              MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0901)
#define DXUTERR_NOCOMPATIBLEDEVICES     MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0902)
#define DXUTERR_MEDIANOTFOUND           MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0903)
#define DXUTERR_NONZEROREFCOUNT         MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0904)
#define DXUTERR_CREATINGDEVICE          MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0905)
#define DXUTERR_RESETTINGDEVICE         MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0906)
#define DXUTERR_CREATINGDEVICEOBJECTS   MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0907)
#define DXUTERR_RESETTINGDEVICEOBJECTS  MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0908)
#define DXUTERR_DEVICEREMOVED           MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x090A)
typedef HRESULT(WINAPI *LPCREATEDXGIFACTORY)(REFIID, void **);
typedef HRESULT(WINAPI *LPD3D10CREATEDEVICE)(IDXGIAdapter *, D3D10_DRIVER_TYPE, HMODULE, UINT, UINT32,
ID3D10Device **);
typedef HRESULT(WINAPI *LPD3D10CREATEDEVICE1)(IDXGIAdapter *, D3D10_DRIVER_TYPE, HMODULE, UINT,
D3D10_FEATURE_LEVEL1, UINT, ID3D10Device1 **);
typedef HRESULT(WINAPI *LPD3D10CREATESTATEBLOCK)(ID3D10Device *pDevice, D3D10_STATE_BLOCK_MASK *pStateBlockMask,
ID3D10StateBlock **ppStateBlock);
typedef HRESULT(WINAPI *LPD3D10STATEBLOCKMASKUNION)(D3D10_STATE_BLOCK_MASK *pA, D3D10_STATE_BLOCK_MASK *pB,
D3D10_STATE_BLOCK_MASK *pResult);
typedef HRESULT(WINAPI *LPD3D10STATEBLOCKMASKINTERSECT)(D3D10_STATE_BLOCK_MASK *pA, D3D10_STATE_BLOCK_MASK *pB,
D3D10_STATE_BLOCK_MASK *pResult);
typedef HRESULT(WINAPI *LPD3D10STATEBLOCKMASKDIFFERENCE)(D3D10_STATE_BLOCK_MASK *pA, D3D10_STATE_BLOCK_MASK *pB,
D3D10_STATE_BLOCK_MASK *pResult);
typedef HRESULT(WINAPI *LPD3D10STATEBLOCKMASKENABLECAPTURE)(D3D10_STATE_BLOCK_MASK *pMask,
D3D10_DEVICE_STATE_TYPES StateType, UINT RangeStart,
UINT RangeLength);
typedef HRESULT(WINAPI *LPD3D10STATEBLOCKMASKDISABLECAPTURE)(D3D10_STATE_BLOCK_MASK *pMask,
D3D10_DEVICE_STATE_TYPES StateType, UINT RangeStart,
UINT RangeLength);
typedef HRESULT(WINAPI *LPD3D10STATEBLOCKMASKENABLEALL)(D3D10_STATE_BLOCK_MASK *pMask);
typedef HRESULT(WINAPI *LPD3D10STATEBLOCKMASKDISABLEALL)(D3D10_STATE_BLOCK_MASK *pMask);
typedef BOOL (WINAPI *LPD3D10STATEBLOCKMASKGETSETTING)(D3D10_STATE_BLOCK_MASK *pMask,
D3D10_DEVICE_STATE_TYPES StateType, UINT Entry);
typedef HRESULT(WINAPI *LPD3D10COMPILEEFFECTFROMMEMORY)(void *pData, SIZE_T DataLength, LPCSTR pSrcFileName,
CONST D3D10_SHADER_MACRO *pDefines,
ID3D10Include *pInclude, UINT HLSLFlags, UINT FXFlags,
ID3D10Blob **ppCompiledEffect, ID3D10Blob **ppErrors);
typedef HRESULT(WINAPI *LPD3D10CREATEEFFECTFROMMEMORY)(void *pData, SIZE_T DataLength, UINT FXFlags,
ID3D10Device *pDevice,
ID3D10EffectPool *pEffectPool,
ID3D10Effect **ppEffect);
typedef HRESULT(WINAPI *LPD3D10CREATEEFFECTPOOLFROMMEMORY)(void *pData, SIZE_T DataLength, UINT FXFlags,
ID3D10Device *pDevice, ID3D10EffectPool **ppEffectPool);
typedef HRESULT(WINAPI *LPD3D10CREATEDEVICEANDSWAPCHAIN)(IDXGIAdapter *pAdapter,
D3D10_DRIVER_TYPE DriverType,
HMODULE Software,
UINT Flags,
UINT SDKVersion,
DXGI_SWAP_CHAIN_DESC *pSwapChainDesc,
IDXGISwapChain **ppSwapChain,
ID3D10Device **ppDevice);
typedef HRESULT(WINAPI *LPD3D10CREATEDEVICEANDSWAPCHAIN1)(IDXGIAdapter *pAdapter,
D3D10_DRIVER_TYPE DriverType,
HMODULE Software,
UINT Flags,
D3D10_FEATURE_LEVEL1 HardwareLevel,
UINT SDKVersion,
DXGI_SWAP_CHAIN_DESC *pSwapChainDesc,
IDXGISwapChain **ppSwapChain,
ID3D10Device1 **ppDevice);
static HMODULE                              g_hModDXGI = NULL;
static HMODULE                              g_hModD3D10 = NULL;
static HMODULE                              g_hModD3D101 = NULL;
static LPCREATEDXGIFACTORY                  sFnPtr_CreateDXGIFactory = NULL;
static LPD3D10CREATESTATEBLOCK              sFnPtr_D3D10CreateStateBlock = NULL;
static LPD3D10CREATEDEVICE                  sFnPtr_D3D10CreateDevice = NULL;
static LPD3D10CREATEDEVICE1                 sFnPtr_D3D10CreateDevice1 = NULL;
static LPD3D10STATEBLOCKMASKUNION           sFnPtr_D3D10StateBlockMaskUnion = NULL;
static LPD3D10STATEBLOCKMASKINTERSECT       sFnPtr_D3D10StateBlockMaskIntersect = NULL;
static LPD3D10STATEBLOCKMASKDIFFERENCE      sFnPtr_D3D10StateBlockMaskDifference = NULL;
static LPD3D10STATEBLOCKMASKENABLECAPTURE   sFnPtr_D3D10StateBlockMaskEnableCapture = NULL;
static LPD3D10STATEBLOCKMASKDISABLECAPTURE  sFnPtr_D3D10StateBlockMaskDisableCapture = NULL;
static LPD3D10STATEBLOCKMASKENABLEALL       sFnPtr_D3D10StateBlockMaskEnableAll = NULL;
static LPD3D10STATEBLOCKMASKDISABLEALL      sFnPtr_D3D10StateBlockMaskDisableAll = NULL;
static LPD3D10STATEBLOCKMASKGETSETTING      sFnPtr_D3D10StateBlockMaskGetSetting = NULL;
static LPD3D10COMPILEEFFECTFROMMEMORY       sFnPtr_D3D10CompileEffectFromMemory = NULL;
static LPD3D10CREATEEFFECTFROMMEMORY        sFnPtr_D3D10CreateEffectFromMemory = NULL;
static LPD3D10CREATEEFFECTPOOLFROMMEMORY    sFnPtr_D3D10CreateEffectPoolFromMemory = NULL;
static LPD3D10CREATEDEVICEANDSWAPCHAIN      sFnPtr_D3D10CreateDeviceAndSwapChain  = NULL;
static LPD3D10CREATEDEVICEANDSWAPCHAIN1     sFnPtr_D3D10CreateDeviceAndSwapChain1 = NULL;
static bool dynlinkUnloadD3D10API(void)
{
if (g_hModD3D10)
{
FreeLibrary(g_hModD3D10);
g_hModD3D10 = NULL;
}
if (g_hModDXGI)
{
FreeLibrary(g_hModDXGI);
g_hModDXGI = NULL;
}
if (g_hModD3D101)
{
FreeLibrary(g_hModD3D101);
g_hModD3D101 = NULL;
}
return true;
}
static bool dynlinkLoadD3D10API(void)
{
g_hModD3D10 = LoadLibrary("d3d10.dll");
if (g_hModD3D10 != NULL)
{
sFnPtr_D3D10CreateStateBlock             = (LPD3D10CREATESTATEBLOCK)           GetProcAddress(g_hModD3D10, "D3D10CreateStateBlock");
sFnPtr_D3D10CreateDevice                 = (LPD3D10CREATEDEVICE)           GetProcAddress(g_hModD3D10, "D3D10CreateDevice");
sFnPtr_D3D10StateBlockMaskUnion          = (LPD3D10STATEBLOCKMASKUNION)        GetProcAddress(g_hModD3D10, "D3D10StateBlockMaskUnion");
sFnPtr_D3D10StateBlockMaskIntersect      = (LPD3D10STATEBLOCKMASKINTERSECT)    GetProcAddress(g_hModD3D10, "D3D10StateBlockMaskIntersect");
sFnPtr_D3D10StateBlockMaskDifference     = (LPD3D10STATEBLOCKMASKDIFFERENCE)   GetProcAddress(g_hModD3D10, "D3D10StateBlockMaskDifference");
sFnPtr_D3D10StateBlockMaskEnableCapture  = (LPD3D10STATEBLOCKMASKENABLECAPTURE) GetProcAddress(g_hModD3D10, "D3D10StateBlockMaskEnableCapture");
sFnPtr_D3D10StateBlockMaskDisableCapture = (LPD3D10STATEBLOCKMASKDISABLECAPTURE)GetProcAddress(g_hModD3D10, "D3D10StateBlockMaskDisableCapture");
sFnPtr_D3D10StateBlockMaskEnableAll      = (LPD3D10STATEBLOCKMASKENABLEALL)    GetProcAddress(g_hModD3D10, "D3D10StateBlockMaskEnableAll");
sFnPtr_D3D10StateBlockMaskDisableAll     = (LPD3D10STATEBLOCKMASKDISABLEALL)   GetProcAddress(g_hModD3D10, "D3D10StateBlockMaskDisableAll");
sFnPtr_D3D10StateBlockMaskGetSetting     = (LPD3D10STATEBLOCKMASKGETSETTING)   GetProcAddress(g_hModD3D10, "D3D10StateBlockMaskGetSetting");
sFnPtr_D3D10CompileEffectFromMemory      = (LPD3D10COMPILEEFFECTFROMMEMORY)    GetProcAddress(g_hModD3D10, "D3D10CompileEffectFromMemory");
sFnPtr_D3D10CreateEffectFromMemory       = (LPD3D10CREATEEFFECTFROMMEMORY)     GetProcAddress(g_hModD3D10, "D3D10CreateEffectFromMemory");
sFnPtr_D3D10CreateEffectPoolFromMemory   = (LPD3D10CREATEEFFECTPOOLFROMMEMORY) GetProcAddress(g_hModD3D10, "D3D10CreateEffectPoolFromMemory");
sFnPtr_D3D10CreateDeviceAndSwapChain     = (LPD3D10CREATEDEVICEANDSWAPCHAIN)    GetProcAddress(g_hModD3D10, "D3D10CreateDeviceAndSwapChain");
}
g_hModDXGI = LoadLibrary("dxgi.dll");
if (g_hModDXGI)
{
sFnPtr_CreateDXGIFactory                 = (LPCREATEDXGIFACTORY)           GetProcAddress(g_hModDXGI , "CreateDXGIFactory");
}
g_hModD3D101 = LoadLibrary("d3d10_1.dll");
if (g_hModD3D101 != NULL)
{
sFnPtr_D3D10CreateDevice1                = (LPD3D10CREATEDEVICE1)              GetProcAddress(g_hModD3D101, "D3D10CreateDevice1");
sFnPtr_D3D10CreateDeviceAndSwapChain1    = (LPD3D10CREATEDEVICEANDSWAPCHAIN1)   GetProcAddress(g_hModD3D101, "D3D10CreateDeviceAndSwapChain1");
}
if (g_hModD3D10 == NULL || g_hModDXGI == NULL || g_hModD3D101 == NULL)
{
dynlinkUnloadD3D10API();
return false;
}
return true;
}
#endif
