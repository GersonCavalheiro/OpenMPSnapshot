



#include "video.h"
#include <sched.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include <AvailabilityMacros.h>
#undef DEPRECATED_ATTRIBUTE
#define DEPRECATED_ATTRIBUTE
#include <Carbon/Carbon.h>
#include <AGL/agl.h>
#include <OpenGL/gl.h>    
#include <OpenGL/glext.h> 

unsigned int *      g_pImg = 0;
int                 g_sizex, g_sizey;
WindowRef           g_window = 0;
static video *      g_video = 0;
static int          g_fps = 0;
struct timeval      g_time;
static pthread_mutex_t  g_mutex = PTHREAD_MUTEX_INITIALIZER;


static OSStatus     AppEventHandler( EventHandlerCallRef inCaller, EventRef inEvent, void* inRefcon );
WindowRef           HandleNew();
static OSStatus     WindowEventHandler( EventHandlerCallRef inCaller, EventRef inEvent, void* inRefcon );

static IBNibRef     sNibRef;


struct structGLInfo 
{
SInt16 width;               
SInt16 height;              
UInt32 pixelDepth;          
Boolean fDepthMust;         
Boolean fAcceleratedMust;   
GLint aglAttributes[64];    
SInt32 VRAM;                
SInt32 textureRAM;          
AGLPixelFormat    fmt;      
};
typedef struct structGLInfo structGLInfo;
typedef struct structGLInfo * pstructGLInfo;

struct structGLWindowInfo 
{
Boolean fAcceleratedMust;   
GLint aglAttributes[64];    
SInt32 VRAM;                
SInt32 textureRAM;          
AGLPixelFormat    fmt;      
Boolean fDraggable;         
};
typedef struct structGLWindowInfo structGLWindowInfo;
typedef struct structGLWindowInfo * pstructGLWindowInfo;


struct recGLCap 
{
Boolean f_ext_texture_rectangle; 
Boolean f_ext_client_storage; 
Boolean f_ext_packed_pixel; 
Boolean f_ext_texture_edge_clamp; 
Boolean f_gl_texture_edge_clamp; 
unsigned long edgeClampParam; 
long maxTextureSize; 
long maxNOPTDTextureSize; 
};
typedef struct recGLCap recGLCap;
typedef recGLCap * pRecGLCap;

struct recImage 
{
structGLWindowInfo glInfo;  
AGLContext aglContext;      
GLuint fontList;            

Boolean fAGPTexturing;      

Boolean fNPOTTextures; 
Boolean fTileTextures; 
Boolean fOverlapTextures; 
Boolean fClientTextures; 

unsigned char * pImageBuffer; 
long imageWidth; 
long imageHeight; 
float imageAspect; 
long imageDepth; 
long textureX; 
long textureY; 
long maxTextureSize; 
GLuint * pTextureName; 
long textureWidth; 
long textureHeight; 
float zoomX; 
float zoomY; 
};
typedef struct recImage recImage; 
typedef recImage * pRecImage; 



OSStatus DestroyGLFromWindow (AGLContext* paglContext, pstructGLWindowInfo pcontextInfo);

short FindGDHandleFromWindow (WindowPtr pWindow, GDHandle * phgdOnThisDevice);

OSStatus DisposeGLForWindow (WindowRef window);

OSStatus BuildGLForWindow (WindowRef window);

OSStatus ResizeMoveGLWindow (WindowRef window);

void DrawGL (WindowRef window);

pRecGLCap gpOpenGLCaps; 


static Boolean CheckRenderer (GDHandle hGD, long *VRAM, long *textureRAM, GLint*  , Boolean fAccelMust);
static Boolean CheckAllDeviceRenderers (long* pVRAM, long* pTextureRAM, GLint* pDepthSizeSupport, Boolean fAccelMust);
static void DumpCurrent (AGLDrawable* paglDraw, AGLContext* paglContext, pstructGLInfo pcontextInfo);
static OSStatus BuildGLonWindow (WindowPtr pWindow, AGLContext* paglContext, pstructGLWindowInfo pcontextInfo, AGLContext aglShareContext);

static long GetNextTextureSize (long textureDimension, long maxTextureSize, Boolean textureRectangle);
static long GetTextureNumFromTextureDim (long textureDimension, long maxTextureSize, Boolean texturesOverlap, Boolean textureRectangle);



#pragma mark -


void ReportErrorNum (char * strError, long numError)
{
char errMsgPStr [257];

errMsgPStr[0] = (char)snprintf (errMsgPStr+1, 255, "%s %ld (0x%lx)\n", strError, numError, numError); 

DebugStr ( (ConstStr255Param) errMsgPStr );
}


void ReportError (char * strError)
{
char errMsgPStr [257];

errMsgPStr[0] = (char)snprintf (errMsgPStr+1, 255, "%s\n", strError); 

DebugStr ( (ConstStr255Param) errMsgPStr );
}



OSStatus aglReportError (void)
{
GLenum err = aglGetError();
if (AGL_NO_ERROR != err)
ReportError ((char *)aglErrorString(err));
if (err == AGL_NO_ERROR)
return noErr;
else
return (OSStatus) err;
}



OSStatus glReportError (void)
{
GLenum err = glGetError();
switch (err)
{
case GL_NO_ERROR:
break;
case GL_INVALID_ENUM:
ReportError ("GL Error: Invalid enumeration");
break;
case GL_INVALID_VALUE:
ReportError ("GL Error: Invalid value");
break;
case GL_INVALID_OPERATION:
ReportError ("GL Error: Invalid operation");
break;
case GL_STACK_OVERFLOW:
ReportError ("GL Error: Stack overflow");
break;
case GL_STACK_UNDERFLOW:
ReportError ("GL Error: Stack underflow");
break;
case GL_OUT_OF_MEMORY:
ReportError ("GL Error: Out of memory");
break;
}
if (err == GL_NO_ERROR)
return noErr;
else
return (OSStatus) err;
}







static Boolean CheckRenderer (GDHandle hGD, long* pVRAM, long* pTextureRAM, GLint* pDepthSizeSupport, Boolean fAccelMust)
{
AGLRendererInfo info, head_info;
GLint inum;
GLint dAccel = 0;
GLint dVRAM = 0, dMaxVRAM = 0;
Boolean canAccel = false, found = false;
head_info = aglQueryRendererInfo(&hGD, 1);
aglReportError ();
if(!head_info)
{
ReportError ("aglQueryRendererInfo error");
return false;
}
else
{
info = head_info;
inum = 0;
while (info)
{    
aglDescribeRenderer(info, AGL_ACCELERATED, &dAccel);
aglReportError ();
if (dAccel)
canAccel = true;
info = aglNextRendererInfo(info);
aglReportError ();
inum++;
}

info = head_info;
inum = 0;
while (info)
{
aglDescribeRenderer (info, AGL_ACCELERATED, &dAccel);
aglReportError ();
if ((canAccel && dAccel) || (!canAccel && (!fAccelMust || dAccel)))
{
aglDescribeRenderer (info, AGL_VIDEO_MEMORY, &dVRAM);    
aglReportError ();
if (dVRAM >= (*pVRAM + *pTextureRAM))
{
if (dVRAM >= dMaxVRAM) 
{
aglDescribeRenderer (info, AGL_DEPTH_MODES, pDepthSizeSupport);    
aglReportError ();
dMaxVRAM = dVRAM; 
found = true;
}
}
}
info = aglNextRendererInfo(info);
aglReportError ();
inum++;
}
}
aglDestroyRendererInfo(head_info);
if (found) 
{
*pVRAM = dMaxVRAM; 
return true;
}
return false;
}






static Boolean CheckAllDeviceRenderers (long* pVRAM, long* pTextureRAM, GLint* pDepthSizeSupport, Boolean fAccelMust)
{
AGLRendererInfo info, head_info;
GLint inum;
GLint dAccel = 0;
GLint dVRAM = 0, dMaxVRAM = 0;
Boolean canAccel = false, found = false, goodCheck = true; 
long MinVRAM = 0x8FFFFFFF; 
GDHandle hGD = GetDeviceList (); 
while (hGD && goodCheck)
{
head_info = aglQueryRendererInfo(&hGD, 1);
aglReportError ();
if(!head_info)
{
ReportError ("aglQueryRendererInfo error");
return false;
}
else
{
info = head_info;
inum = 0;
while (info)
{
aglDescribeRenderer(info, AGL_ACCELERATED, &dAccel);
aglReportError ();
if (dAccel)
canAccel = true;
info = aglNextRendererInfo(info);
aglReportError ();
inum++;
}

info = head_info;
inum = 0;
while (info)
{    
aglDescribeRenderer(info, AGL_ACCELERATED, &dAccel);
aglReportError ();
if ((canAccel && dAccel) || (!canAccel && (!fAccelMust || dAccel)))
{
aglDescribeRenderer(info, AGL_VIDEO_MEMORY, &dVRAM);    
aglReportError ();
if (dVRAM >= (*pVRAM + *pTextureRAM))
{
if (dVRAM >= dMaxVRAM) 
{
aglDescribeRenderer(info, AGL_DEPTH_MODES, pDepthSizeSupport);    
aglReportError ();
dMaxVRAM = dVRAM; 
found = true;
}
}
}
info = aglNextRendererInfo(info);
aglReportError ();
inum++;
}
}
aglDestroyRendererInfo(head_info);
if (found) 
{
if (MinVRAM > dMaxVRAM)
MinVRAM = dMaxVRAM; 

}
else
goodCheck = false; 
hGD = GetNextDevice (hGD); 
} 
if (goodCheck) 
{
*pVRAM = MinVRAM; 
return true;
}
return false; 
}





void DumpCurrent (AGLDrawable* paglDraw, AGLContext* paglContext, pstructGLInfo pcontextInfo)
{
if (*paglContext)
{
aglSetCurrentContext (NULL);
aglReportError ();
aglSetDrawable (*paglContext, NULL);
aglReportError ();
aglDestroyContext (*paglContext);
aglReportError ();
*paglContext = NULL;
}

if (pcontextInfo->fmt)
{
aglDestroyPixelFormat (pcontextInfo->fmt); 
aglReportError ();
}
pcontextInfo->fmt = 0;

if (*paglDraw) 
DisposeWindow (GetWindowFromPort (*paglDraw));
*paglDraw = NULL;
}

#pragma mark -


static OSStatus BuildGLonWindow (WindowPtr pWindow, AGLContext* paglContext, pstructGLWindowInfo pcontextInfo, AGLContext aglShareContext)
{
GDHandle hGD = NULL;
GrafPtr cgrafSave = NULL;
short numDevices;
GLint depthSizeSupport;
OSStatus err = noErr;

if (!pWindow || !pcontextInfo)
{
ReportError ("NULL parameter passed to BuildGLonWindow.");
return paramErr;
}

GetPort (&cgrafSave);
SetPortWindowPort(pWindow);

numDevices = FindGDHandleFromWindow (pWindow, &hGD);
if (!pcontextInfo->fDraggable)     
{
if ((numDevices > 1) || (numDevices == 0)) 
{
if (pcontextInfo->fAcceleratedMust)
{
ReportError ("Unable to accelerate window that spans multiple devices");
return err;
}
}
else 
{
if (!CheckRenderer (hGD, &(pcontextInfo->VRAM), &(pcontextInfo->textureRAM), &depthSizeSupport, pcontextInfo->fAcceleratedMust))
{
ReportError ("Renderer check failed");
return err;
}
}
}
else if (!CheckAllDeviceRenderers (&(pcontextInfo->VRAM), &(pcontextInfo->textureRAM), &depthSizeSupport, pcontextInfo->fAcceleratedMust))
{
ReportError ("Renderer check failed");
return err;
}

if ((Ptr) kUnresolvedCFragSymbolAddress == (Ptr) aglChoosePixelFormat) 
{
ReportError ("OpenGL not installed");
return noErr;
}    

if ((!pcontextInfo->fDraggable && (numDevices == 1)))  
pcontextInfo->fmt = aglChoosePixelFormat (&hGD, 1, pcontextInfo->aglAttributes); 
else
pcontextInfo->fmt = aglChoosePixelFormat (NULL, 0, pcontextInfo->aglAttributes); 
aglReportError ();
if (NULL == pcontextInfo->fmt) 
{
ReportError("Could not find valid pixel format");
return noErr;
}

*paglContext = aglCreateContext (pcontextInfo->fmt, aglShareContext); 
if (AGL_BAD_MATCH == aglGetError())
*paglContext = aglCreateContext (pcontextInfo->fmt, 0); 
aglReportError ();
if (NULL == *paglContext) 
{
ReportError ("Could not create context");
return noErr;
}

if (!aglSetDrawable (*paglContext, GetWindowPort (pWindow))) 
return aglReportError ();

if(!aglSetCurrentContext (*paglContext)) 
return aglReportError ();

SetPort (cgrafSave);

return err;
}

#pragma mark -




OSStatus DestroyGLFromWindow (AGLContext* paglContext, pstructGLWindowInfo pcontextInfo)
{
OSStatus err;

if ((!paglContext) || (!*paglContext))
return paramErr; 
glFinish ();
aglSetCurrentContext (NULL);
err = aglReportError ();
aglSetDrawable (*paglContext, NULL);
err = aglReportError ();
aglDestroyContext (*paglContext);
err = aglReportError ();
*paglContext = NULL;

if (pcontextInfo->fmt)
{
aglDestroyPixelFormat (pcontextInfo->fmt); 
err = aglReportError ();
}
pcontextInfo->fmt = 0;

return err;
}






short FindGDHandleFromWindow (WindowPtr pWindow, GDHandle * phgdOnThisDevice)
{
GrafPtr pgpSave;
Rect rectWind, rectSect;
long greatestArea, sectArea;
short numDevices = 0;
GDHandle hgdNthDevice;

if (!pWindow || !phgdOnThisDevice)
return 0;

*phgdOnThisDevice = NULL;

GetPort (&pgpSave);
SetPortWindowPort (pWindow);


GetWindowPortBounds (pWindow, &rectWind);
LocalToGlobal ((Point*)& rectWind.top);    
LocalToGlobal ((Point*)& rectWind.bottom);
hgdNthDevice = GetDeviceList ();
greatestArea = 0;
while (hgdNthDevice)
{
if (TestDeviceAttribute (hgdNthDevice, screenDevice))
if (TestDeviceAttribute (hgdNthDevice, screenActive))
{
SectRect (&rectWind, &(**hgdNthDevice).gdRect, &rectSect);
sectArea = (long) (rectSect.right - rectSect.left) * (rectSect.bottom - rectSect.top);
if (sectArea > 0)
numDevices++;
if (sectArea > greatestArea)
{
greatestArea = sectArea; 
*phgdOnThisDevice = hgdNthDevice; 
}
hgdNthDevice = GetNextDevice(hgdNthDevice);
}
}

SetPort (pgpSave);
return numDevices;
}



static long GetNextTextureSize (long textureDimension, long maxTextureSize, Boolean textureRectangle)
{
long targetTextureSize = maxTextureSize; 
if (textureRectangle)
{
if (textureDimension >= targetTextureSize) 
return targetTextureSize; 
else
return textureDimension; 
}
else
{
do 
{  
if (textureDimension >= targetTextureSize) 
return targetTextureSize; 
}
while (targetTextureSize >>= 1); 
}
return 0; 
}



static long GetTextureNumFromTextureDim (long textureDimension, long maxTextureSize, Boolean texturesOverlap, Boolean textureRectangle) 
{

long i = 0; 
long bitValue = maxTextureSize; 
long texOverlapx2 = texturesOverlap ? 2 : 0;
textureDimension -= texOverlapx2; 
if (textureRectangle)
{
while (textureDimension > (bitValue - texOverlapx2)) 
{
i++; 
textureDimension -= bitValue - texOverlapx2; 
}
i++; 
}
else
{
do
{
while (textureDimension >= (bitValue - texOverlapx2)) 
{
i++; 
textureDimension -= bitValue - texOverlapx2; 
}
}
while ((bitValue >>= 1) > texOverlapx2); 
if (textureDimension > 0x0) 
ReportErrorNum ("GetTextureNumFromTextureDim error: Texture to small to draw, should not ever get here, texture size remaining:", textureDimension);
}
return i; 
} 

#pragma mark -


OSStatus DisposeGLForWindow (WindowRef window)
{
if (window)
{
pRecImage pWindowInfo = (pRecImage) GetWRefCon (window); 
SetWRefCon (window, 0); 
if (NULL == pWindowInfo) 
return paramErr; 
if (NULL != pWindowInfo->aglContext)
{
aglSetCurrentContext (pWindowInfo->aglContext); 
aglUpdateContext (pWindowInfo->aglContext); 
glFinish (); 
glDeleteTextures (pWindowInfo->textureX * pWindowInfo->textureY, pWindowInfo->pTextureName); 
DestroyGLFromWindow (&pWindowInfo->aglContext, &pWindowInfo->glInfo); 
pWindowInfo->aglContext = NULL; 
}
if (NULL != pWindowInfo->pTextureName)
{
DisposePtr ((Ptr) pWindowInfo->pTextureName); 
pWindowInfo->pTextureName = NULL; 
}
if (pWindowInfo->pImageBuffer) 
{
pWindowInfo->pImageBuffer = NULL;
}
DisposePtr ((Ptr) pWindowInfo);
return noErr; 
}
else
return paramErr; 
}



OSStatus BuildGLForWindow (WindowRef window)
{
GrafPtr portSave = NULL; 
pRecImage pWindowInfo = (pRecImage) GetWRefCon (window); 
short i; 
GLenum textureTarget = GL_TEXTURE_2D;

if (!pWindowInfo->aglContext) 
{
GetPort (&portSave);    
SetPort ((GrafPtr) GetWindowPort (window)); 
pWindowInfo->glInfo.fAcceleratedMust = false; 
pWindowInfo->glInfo.VRAM = 0 * 1048576; 
pWindowInfo->glInfo.textureRAM = 0 * 1048576; 
pWindowInfo->glInfo.fDraggable = true; 
pWindowInfo->glInfo.fmt = 0; 

i = 0; 
pWindowInfo->glInfo.aglAttributes [i++] = AGL_RGBA; 
pWindowInfo->glInfo.aglAttributes [i++] = AGL_DOUBLEBUFFER; 
pWindowInfo->glInfo.aglAttributes [i++] = AGL_ACCELERATED; 
pWindowInfo->glInfo.aglAttributes [i++] = AGL_NO_RECOVERY; 
pWindowInfo->glInfo.aglAttributes [i++] = AGL_NONE; 
BuildGLonWindow (window, &(pWindowInfo->aglContext), &(pWindowInfo->glInfo), NULL); 
if (!pWindowInfo->aglContext) 
DestroyGLFromWindow (&pWindowInfo->aglContext, &pWindowInfo->glInfo); 
else 
{
GLint swap = 0; 
Rect rectPort; 
long width = pWindowInfo->imageWidth, height = pWindowInfo->imageHeight; 
GDHandle device; 
Rect deviceRect, availRect, rect; 

GetWindowGreatestAreaDevice (window, kWindowContentRgn, &device, &deviceRect); 
GetAvailableWindowPositioningBounds (device, &availRect); 
if (width > (availRect.right - availRect.left)) 
width = (availRect.right - availRect.left);
if (height > (availRect.bottom - availRect.top)) 
height = (availRect.bottom - availRect.top);
SizeWindow (window, (short) width, (short) height, true); 
ConstrainWindowToScreen(window, kWindowStructureRgn, kWindowConstrainMayResize, NULL, &rect); 
GetWindowPortBounds (window, &rectPort); 

aglSetCurrentContext (pWindowInfo->aglContext); 
aglUpdateContext (pWindowInfo->aglContext); 
InvalWindowRect (window, &rectPort); 
glViewport (0, 0, rectPort.right - rectPort.left, rectPort.bottom - rectPort.top); 

aglSetInteger (pWindowInfo->aglContext, AGL_SWAP_INTERVAL, &swap); 

#ifdef GL_TEXTURE_RECTANGLE_EXT
if (pWindowInfo->fNPOTTextures)
textureTarget = GL_TEXTURE_RECTANGLE_EXT;
#endif

glEnable (textureTarget); 

glClearColor(0.0f, 0.0f, 0.0f, 1.0f); 
glClear (GL_COLOR_BUFFER_BIT); 
aglSwapBuffers (pWindowInfo->aglContext); 


#ifdef GL_TEXTURE_RECTANGLE_EXT
if (pWindowInfo->fNPOTTextures)
glEnable(GL_TEXTURE_RECTANGLE_EXT);
#endif
if (pWindowInfo->fAGPTexturing)
glTextureRangeAPPLE(textureTarget, pWindowInfo->textureHeight * pWindowInfo->textureWidth * (pWindowInfo->imageDepth >> 3), pWindowInfo->pImageBuffer);
glPixelStorei (GL_UNPACK_ROW_LENGTH, pWindowInfo->textureWidth); 
pWindowInfo->textureX = GetTextureNumFromTextureDim (pWindowInfo->textureWidth, pWindowInfo->maxTextureSize, pWindowInfo->fOverlapTextures, pWindowInfo->fNPOTTextures); 
pWindowInfo->textureY = GetTextureNumFromTextureDim (pWindowInfo->textureHeight, pWindowInfo->maxTextureSize, pWindowInfo->fOverlapTextures, pWindowInfo->fNPOTTextures); 
pWindowInfo->pTextureName = (GLuint *) NewPtrClear ((long) sizeof (GLuint) * pWindowInfo->textureX * pWindowInfo->textureY); 
glGenTextures (pWindowInfo->textureX * pWindowInfo->textureY, pWindowInfo->pTextureName); 
{
long x, y, k = 0, offsetY, offsetX = 0, currWidth, currHeight; 
for (x = 0; x < pWindowInfo->textureX; x++) 
{
currWidth = GetNextTextureSize (pWindowInfo->textureWidth - offsetX, pWindowInfo->maxTextureSize, pWindowInfo->fNPOTTextures); 
offsetY = 0; 
for (y = 0; y < pWindowInfo->textureY; y++) 
{
unsigned char * pBuffer = pWindowInfo->pImageBuffer + 
offsetY * pWindowInfo->textureWidth * (pWindowInfo->imageDepth >> 3) + 
offsetX * (pWindowInfo->imageDepth >> 3);
currHeight = GetNextTextureSize (pWindowInfo->textureHeight - offsetY, pWindowInfo->maxTextureSize, pWindowInfo->fNPOTTextures); 
glBindTexture (textureTarget, pWindowInfo->pTextureName[k++]);
if (pWindowInfo->fAGPTexturing) {
glTexParameterf (textureTarget, GL_TEXTURE_PRIORITY, 0.0f); 
glTexParameteri (textureTarget, GL_TEXTURE_STORAGE_HINT_APPLE, GL_STORAGE_SHARED_APPLE);
}
else
glTexParameterf (textureTarget, GL_TEXTURE_PRIORITY, 1.0f);

#ifdef GL_UNPACK_CLIENT_STORAGE_APPLE
if (pWindowInfo->fClientTextures)
glPixelStorei (GL_UNPACK_CLIENT_STORAGE_APPLE, 1);
else
glPixelStorei (GL_UNPACK_CLIENT_STORAGE_APPLE, 0);
#endif

glTexParameteri (textureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexParameteri (textureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
glTexParameteri (textureTarget, GL_TEXTURE_WRAP_S, gpOpenGLCaps->edgeClampParam);
glTexParameteri (textureTarget, GL_TEXTURE_WRAP_T, gpOpenGLCaps->edgeClampParam);
glReportError (); 
glTexImage2D (textureTarget, 0, GL_RGBA, currWidth, currHeight, 0, 
GL_BGRA_EXT, pWindowInfo->imageDepth == 32 ? GL_UNSIGNED_INT_8_8_8_8_REV : GL_UNSIGNED_SHORT_1_5_5_5_REV, 
pBuffer); 
glReportError (); 
offsetY += currHeight - 2 * pWindowInfo->fOverlapTextures; 
}
offsetX += currWidth - 2 * pWindowInfo->fOverlapTextures; 
}
}
if (false == pWindowInfo->fClientTextures) 
{
DisposePtr ((Ptr) pWindowInfo->pImageBuffer); 
pWindowInfo->pImageBuffer = NULL;
}
}
SetPort (portSave); 
}
return noErr; 
}



OSStatus ResizeMoveGLWindow (WindowRef window)
{
OSStatus err = noErr; 
Rect rectPort; 
pRecImage pWindowInfo = (pRecImage) GetWRefCon (window); 
if (window && pWindowInfo) 
{
GetWindowPortBounds (window, &rectPort);
pWindowInfo->zoomX = (float) (rectPort.right - rectPort.left) / (float) pWindowInfo->imageWidth;
pWindowInfo->zoomY = (float) (rectPort.bottom - rectPort.top) / (float) pWindowInfo->imageHeight;

if (!aglUpdateContext (pWindowInfo->aglContext)) 
aglReportError (); 
if (noErr != err)
ReportErrorNum ("ResizeMoveGLWindow error with InvalWindowRect on window: ", err);  
err = InvalWindowRect (window, &rectPort);
}
else
err = paramErr; 
return err; 
}



void DrawGL (WindowRef window)
{
Rect rectPort; 
pRecImage pWindowInfo; 
long width, height; 
long effectiveTextureMod = 0; 
long x, y, k = 0, offsetY, offsetX = 0, currTextureWidth, currTextureHeight;
GLenum textureTarget = GL_TEXTURE_2D;

if (NULL == window) 
return; 
pWindowInfo = (pRecImage) GetWRefCon (window); 
if (NULL == pWindowInfo) 
return; 
if (NULL == pWindowInfo->aglContext) 
BuildGLForWindow (window);
if (NULL == pWindowInfo->aglContext) 
return;

if (pWindowInfo->fOverlapTextures)
effectiveTextureMod = 2; 
#ifdef GL_TEXTURE_RECTANGLE_EXT
if (pWindowInfo->fNPOTTextures)
textureTarget = GL_TEXTURE_RECTANGLE_EXT;
#endif

aglSetCurrentContext (pWindowInfo->aglContext); 
aglUpdateContext (pWindowInfo->aglContext); 

GetWindowPortBounds (window, &rectPort); 
width = rectPort.right - rectPort.left; 
height = rectPort.bottom - rectPort.top; 
glViewport (0, 0, width, height); 

glMatrixMode (GL_PROJECTION); 
glLoadIdentity (); 
glMatrixMode (GL_MODELVIEW); 
glLoadIdentity (); 
glReportError (); 

glScalef (2.0f / width, -2.0f /  height, 1.0f); 
glReportError (); 

glClear (GL_COLOR_BUFFER_BIT); 

glEnable (textureTarget); 
glColor3f (1.0f, 1.0f, 1.0f); 
for (x = 0; x < pWindowInfo->textureX; x++) 
{
currTextureWidth = GetNextTextureSize (pWindowInfo->textureWidth - offsetX, pWindowInfo->maxTextureSize, pWindowInfo->fNPOTTextures) - effectiveTextureMod; 
offsetY = 0; 
for (y = 0; y < pWindowInfo->textureY; y++) 
{
currTextureHeight = GetNextTextureSize (pWindowInfo->textureHeight - offsetY, pWindowInfo->maxTextureSize, pWindowInfo->fNPOTTextures) - effectiveTextureMod; 
glBindTexture(textureTarget, pWindowInfo->pTextureName[k++]); 
if (!pWindowInfo->fAGPTexturing)
glTexSubImage2D(textureTarget, 0, 0, 0, currTextureWidth, currTextureHeight, GL_BGRA, pWindowInfo->imageDepth == 32 ? GL_UNSIGNED_INT_8_8_8_8_REV : GL_UNSIGNED_SHORT_1_5_5_5_REV, pWindowInfo->pImageBuffer);
glReportError (); 
{
float endX = pWindowInfo->fTileTextures ? currTextureWidth + offsetX : pWindowInfo->imageWidth;
float endY = pWindowInfo->fTileTextures ? currTextureHeight + offsetY : pWindowInfo->imageHeight;
float startXDraw = (offsetX - pWindowInfo->imageWidth * 0.5f) * pWindowInfo->zoomX; 
float endXDraw = (endX - pWindowInfo->imageWidth * 0.5f) * pWindowInfo->zoomX; 
float startYDraw = (offsetY - pWindowInfo->imageHeight * 0.5f) * pWindowInfo->zoomY; 
float endYDraw = (endY - pWindowInfo->imageHeight * 0.5f) * pWindowInfo->zoomY; 
float texOverlap =  pWindowInfo->fOverlapTextures ? 1.0f : 0.0f; 
float startXTexCoord = texOverlap / (currTextureWidth + 2.0f * texOverlap); 
float endXTexCoord = 1.0f - startXTexCoord; 
float startYTexCoord = texOverlap / (currTextureHeight + 2.0f * texOverlap); 
float endYTexCoord = 1.0f - startYTexCoord; 
if (pWindowInfo->fNPOTTextures)
{
startXTexCoord = texOverlap; 
endXTexCoord = currTextureWidth + texOverlap; 
startYTexCoord = texOverlap; 
endYTexCoord = currTextureHeight + texOverlap; 
}
if (endX > (pWindowInfo->imageWidth + 0.5)) 
{
endXDraw = (pWindowInfo->imageWidth * 0.5f) * pWindowInfo->zoomX; 
if (pWindowInfo->fNPOTTextures)
endXTexCoord -= 1.0f;
else
endXTexCoord = 1.0f -  2.0f * startXTexCoord; 
}
if (endY > (pWindowInfo->imageHeight + 0.5f)) 
{
endYDraw = (pWindowInfo->imageHeight * 0.5f) * pWindowInfo->zoomY; 
if (pWindowInfo->fNPOTTextures)
endYTexCoord -= 1.0f;
else
endYTexCoord = 1.0f -  2.0f * startYTexCoord; 
}

glBegin (GL_TRIANGLE_STRIP); 
glTexCoord2f (startXTexCoord, startYTexCoord); 
glVertex3d (startXDraw, startYDraw, 0.0);

glTexCoord2f (endXTexCoord, startYTexCoord); 
glVertex3d (endXDraw, startYDraw, 0.0);

glTexCoord2f (startXTexCoord, endYTexCoord); 
glVertex3d (startXDraw, endYDraw, 0.0);

glTexCoord2f (endXTexCoord, endYTexCoord); 
glVertex3d (endXDraw, endYDraw, 0.0);
glEnd();

}


glReportError (); 
offsetY += currTextureHeight; 
}
offsetX += currTextureWidth; 
}
glReportError (); 

glDisable (textureTarget); 

aglSwapBuffers (pWindowInfo->aglContext);
}


static void FindMinimumOpenGLCapabilities (pRecGLCap pOpenGLCaps)
{
WindowPtr pWin = NULL; 
Rect rectWin = {0, 0, 10, 10};
GLint attrib[] = { AGL_RGBA, AGL_NONE };
AGLPixelFormat fmt = NULL;
AGLContext ctx = NULL;
GLint deviceMaxTextureSize = 0, NPOTDMaxTextureSize = 0;

if (NULL != gpOpenGLCaps)
{
pOpenGLCaps->f_ext_texture_rectangle = true;
pOpenGLCaps->f_ext_client_storage = true;
pOpenGLCaps->f_ext_packed_pixel = true;
pOpenGLCaps->f_ext_texture_edge_clamp = true;
pOpenGLCaps->f_gl_texture_edge_clamp = true;
pOpenGLCaps->maxTextureSize = 0x7FFFFFFF;
pOpenGLCaps->maxNOPTDTextureSize = 0x7FFFFFFF;

pWin = NewCWindow (0L, &rectWin, NULL, false,
plainDBox, (WindowPtr) -1L, true, 0L);

fmt = aglChoosePixelFormat(NULL, 0, attrib);
if (fmt)
ctx = aglCreateContext(fmt, NULL);
if (ctx)
{
GDHandle hgdNthDevice;

aglSetDrawable(ctx, GetWindowPort (pWin));
aglSetCurrentContext(ctx);

hgdNthDevice = GetDeviceList ();
while (hgdNthDevice)
{
if (TestDeviceAttribute (hgdNthDevice, screenDevice))
if (TestDeviceAttribute (hgdNthDevice, screenActive))
{
MoveWindow (pWin, (**hgdNthDevice).gdRect.left + 5, (**hgdNthDevice).gdRect.top + 5, false);
aglUpdateContext(ctx);

{
enum { kShortVersionLength = 32 };
const GLubyte * strVersion = glGetString (GL_VERSION); 
const GLubyte * strExtension = glGetString (GL_EXTENSIONS);    

GLubyte strShortVersion [kShortVersionLength];
short i = 0;
while ((((strVersion[i] <= '9') && (strVersion[i] >= '0')) || (strVersion[i] == '.')) && (i < kShortVersionLength)) 
strShortVersion [i] = strVersion[i++];
strShortVersion [i] = 0; 

pOpenGLCaps->f_ext_texture_rectangle = 
pOpenGLCaps->f_ext_texture_rectangle && (NULL != strstr ((const char *) strExtension, "GL_EXT_texture_rectangle"));
pOpenGLCaps->f_ext_client_storage = 
pOpenGLCaps->f_ext_client_storage && (NULL != strstr ((const char *) strExtension, "GL_APPLE_client_storage"));
pOpenGLCaps->f_ext_packed_pixel = 
pOpenGLCaps->f_ext_packed_pixel && (NULL != strstr ((const char *) strExtension, "GL_APPLE_packed_pixel"));
pOpenGLCaps->f_ext_texture_edge_clamp = 
pOpenGLCaps->f_ext_texture_edge_clamp && (NULL != strstr ((const char *) strExtension, "GL_SGIS_texture_edge_clamp"));
pOpenGLCaps->f_gl_texture_edge_clamp = 
pOpenGLCaps->f_gl_texture_edge_clamp && (!strstr ((const char *) strShortVersion, "1.0") && !strstr ((const char *) strShortVersion, "1.1")); 

glGetIntegerv (GL_MAX_TEXTURE_SIZE, &deviceMaxTextureSize);
if (deviceMaxTextureSize < pOpenGLCaps->maxTextureSize)
pOpenGLCaps->maxTextureSize = deviceMaxTextureSize;
if (NULL != strstr ((const char *) strExtension, "GL_EXT_texture_rectangle"))
{
#ifdef GL_MAX_RECTANGLE_TEXTURE_SIZE_EXT
glGetIntegerv (GL_MAX_RECTANGLE_TEXTURE_SIZE_EXT, &NPOTDMaxTextureSize);
if (NPOTDMaxTextureSize < pOpenGLCaps->maxNOPTDTextureSize)
pOpenGLCaps->maxNOPTDTextureSize = NPOTDMaxTextureSize;
#endif
}
}
hgdNthDevice = GetNextDevice(hgdNthDevice);
}
}
aglDestroyContext( ctx );
}
else
{ 
pOpenGLCaps->f_ext_texture_rectangle = false;
pOpenGLCaps->f_ext_client_storage = false;
pOpenGLCaps->f_ext_packed_pixel = false;
pOpenGLCaps->f_ext_texture_edge_clamp = false;
pOpenGLCaps->f_gl_texture_edge_clamp = false;
pOpenGLCaps->maxTextureSize = 0;
}

if (pOpenGLCaps->f_gl_texture_edge_clamp) 
pOpenGLCaps->edgeClampParam = GL_CLAMP_TO_EDGE;  
else if (pOpenGLCaps->f_ext_texture_edge_clamp) 
pOpenGLCaps->edgeClampParam = GL_CLAMP_TO_EDGE_SGIS; 
else
pOpenGLCaps->edgeClampParam = GL_CLAMP; 

aglDestroyPixelFormat( fmt );
DisposeWindow( pWin );
}
}


static OSStatus
WindowEventHandler( EventHandlerCallRef inCaller, EventRef inEvent, void* inRefcon )
{
OSStatus    err = eventNotHandledErr;
WindowRef    window = (WindowRef) inRefcon;

if( GetEventClass(inEvent) == kEventClassMouse )
{
Point mousePoint; 
verify_noerr( GetEventParameter(inEvent, kEventParamMouseLocation, typeQDPoint, NULL, sizeof(Point), NULL, &mousePoint) );
pRecImage pWindowInfo = (pRecImage) GetWRefCon (window); 
if(pWindowInfo) {
SetPortWindowPort(window);
GlobalToLocal (&mousePoint); 
mousePoint.h /= pWindowInfo->zoomX; mousePoint.v /= pWindowInfo->zoomY;
if(mousePoint.h >= 0 && mousePoint.h < pWindowInfo->imageWidth && mousePoint.v >= 0 && mousePoint.v < pWindowInfo->imageHeight)
g_video->on_mouse(mousePoint.h, mousePoint.v, GetEventKind(inEvent) == kEventMouseUp?-1:1), err = noErr;
}
}
else if( GetEventClass(inEvent) == kEventClassKeyboard )
{
char ch;
verify_noerr( GetEventParameter( inEvent, kEventParamKeyMacCharCodes, typeChar, NULL, sizeof( ch ), NULL, &ch ) );
if(g_video)
g_video->on_key(ch);
}
else 
{
if (GetEventKind(inEvent) == kEventWindowDrawContent)
{
err = noErr;
}
else if (GetEventKind(inEvent) == kEventWindowClose)
{
if (window)
{
g_video->running = false;
}
err = noErr;
}
else if (GetEventKind(inEvent) == kEventWindowShowing)
{
err = BuildGLForWindow (window);
}
else if ((GetEventKind(inEvent) == kEventWindowResizeCompleted) || (GetEventKind(inEvent) == kEventWindowDragCompleted))
{
err = ResizeMoveGLWindow (window);
}
else if (GetEventKind(inEvent) == kEventWindowZoomed)
{
err = ResizeMoveGLWindow (window);
}
}

return err;
}
DEFINE_ONE_SHOT_HANDLER_GETTER( WindowEventHandler )

WindowRef HandleNew()
{
OSStatus  err;
WindowRef window;
pRecImage pWindowInfo = NULL;
static const EventTypeSpec    kWindowEvents[] =
{
{ kEventClassMouse, kEventMouseUp },
{ kEventClassMouse, kEventMouseDown },
{ kEventClassKeyboard, kEventRawKeyDown },
{ kEventClassWindow, kEventWindowShowing },
{ kEventClassWindow, kEventWindowClose },
{ kEventClassWindow, kEventWindowDrawContent },
{ kEventClassWindow, kEventWindowResizeCompleted },
{ kEventClassWindow, kEventWindowDragCompleted },
{ kEventClassWindow, kEventWindowZoomed}
};
if (!gpOpenGLCaps)
{
gpOpenGLCaps = (pRecGLCap) NewPtrClear (sizeof (recGLCap));
FindMinimumOpenGLCapabilities (gpOpenGLCaps);
}

err = CreateWindowFromNib( sNibRef, CFSTR("MainWindow"), &window );
require_noerr( err, CantCreateWindow );
DisposeNibReference(sNibRef);

pWindowInfo = (recImage *) NewPtrClear (sizeof (recImage));
pWindowInfo->textureWidth = pWindowInfo->imageWidth = g_sizex;
pWindowInfo->textureHeight = pWindowInfo->imageHeight = g_sizey;
pWindowInfo->imageDepth = 32;
pWindowInfo->fTileTextures = true;
pWindowInfo->fOverlapTextures = false; 
pWindowInfo->maxTextureSize = gpOpenGLCaps->maxTextureSize;
pWindowInfo->fNPOTTextures = gpOpenGLCaps->f_ext_texture_rectangle;
pWindowInfo->fClientTextures = gpOpenGLCaps->f_ext_client_storage; 
pWindowInfo->fAGPTexturing = true; 
pWindowInfo->pImageBuffer = (unsigned char*) g_pImg;
pWindowInfo->zoomX = 1.0f; 
pWindowInfo->zoomY = 1.0f; 
SetWRefCon (window, (long) pWindowInfo);
char buffer[256]; buffer[0] = snprintf(buffer+1, 255, "%s", g_video->title);
SetWTitle (window, (ConstStr255Param)buffer);
InstallStandardEventHandler(GetWindowEventTarget(window));
InstallWindowEventHandler( window, GetWindowEventHandlerUPP(),
GetEventTypeCount( kWindowEvents ), kWindowEvents, window, NULL );
if (noErr != BuildGLForWindow (window))
{
DisposeGLForWindow (window);
DisposeWindow (window);
return 0;
}

RepositionWindow( window, NULL, kWindowCascadeOnMainScreen );

ShowWindow( window );
return window;

CantCreateWindow:
return 0;
}


static OSStatus
AppEventHandler( EventHandlerCallRef inCaller, EventRef inEvent, void* inRefcon )
{
OSStatus    result = eventNotHandledErr;

return result;
}


video::video()
: red_mask(0xff0000), red_shift(16), green_mask(0xff00),
green_shift(8), blue_mask(0xff), blue_shift(0), depth(24)
{
assert(g_video == 0);
g_video = this; title = "Video"; updating = true; calc_fps = false;
}

bool video::init_window(int x, int y)
{
g_sizex = x; g_sizey = y; g_window = 0;
g_pImg = new unsigned int[x*y];

if( CGGetOnlineDisplayList(0, NULL, NULL) ) {
running = true; 
return false;
}

OSStatus                    err;
static const EventTypeSpec    kAppEvents[] =
{
{ kEventClassCommand, kEventCommandProcess }
};

err = CreateNibReference( CFSTR("main"), &sNibRef );
require_noerr( err, ReturnLabel );

InstallStandardEventHandler(GetApplicationEventTarget()); 
verify_noerr( InstallApplicationEventHandler( NewEventHandlerUPP( AppEventHandler ),
GetEventTypeCount( kAppEvents ), kAppEvents, 0, NULL ) );

InstallStandardEventHandler(GetMenuEventTarget(AcquireRootMenu()));

g_window = HandleNew();

ReturnLabel:
return running = g_window != 0;
}

bool video::init_console()
{
running = true;
return true;
}

void video::terminate()
{
g_video = 0; running = false;
if(g_pImg) { delete[] g_pImg; g_pImg = 0; }
if(g_window) {
DisposeGLForWindow (g_window);
DisposeWindow (g_window);
g_window = 0;
}
}

video::~video()
{
if(g_video) terminate();
}

bool video::next_frame()
{
if(!running) return false;
if(!g_window) return running;
if(threaded && pthread_mutex_trylock(&g_mutex))
return running;
g_fps++;
struct timezone tz; struct timeval now_time; gettimeofday(&now_time, &tz);
double sec = (now_time.tv_sec+1.0*now_time.tv_usec/1000000.0) - (g_time.tv_sec+1.0*g_time.tv_usec/1000000.0);
if(sec > 1) {
memcpy(&g_time, &now_time, sizeof(g_time));
if(calc_fps) {
double fps = g_fps; g_fps = 0;
char buffer[256]; buffer[0] = snprintf(buffer+1, 255, "%s%s: %d fps", title, updating?"":" (no updating)", int(fps/sec));
SetWTitle (g_window, (ConstStr255Param) buffer );
}
}

EventRef theEvent;
EventTargetRef theTarget;
OSStatus                    err;

theTarget = GetEventDispatcherTarget();
while( (err = ReceiveNextEvent(0, NULL, kEventDurationNoWait, true, &theEvent)) == noErr)
{
SendEventToEventTarget(theEvent, theTarget);
ReleaseEvent(theEvent);
}
if(err != eventLoopTimedOutErr) running = false;
if(updating) {
pRecImage pWindowInfo = (pRecImage) GetWRefCon (g_window); 
if(pWindowInfo) DrawGL(g_window);
}
if(threaded) pthread_mutex_unlock(&g_mutex);
return true;
}

void video::main_loop()
{
struct timezone tz; gettimeofday(&g_time, &tz);
on_process();
}

void video::show_title()
{
char buffer[256]; buffer[0] = snprintf(buffer+1, 255, "%s", title);
SetWTitle (g_window, (ConstStr255Param) buffer );
}


drawing_area::drawing_area(int x, int y, int sizex, int sizey)
: start_x(x), start_y(y), size_x(sizex), size_y(sizey), pixel_depth(24),
base_index(y*g_sizex + x), max_index(g_sizex*g_sizey), index_stride(g_sizex), ptr32(g_pImg)
{
assert(x < g_sizex); assert(y < g_sizey);
assert(x+sizex <= g_sizex); assert(y+sizey <= g_sizey);

index = base_index; 
}

drawing_area::~drawing_area() {}
