#ifndef __glut_h__
#define __glut_h__





#if defined(_WIN32)


# if 0

#  define  WIN32_LEAN_AND_MEAN
#  include <windows.h>
# else

#  ifndef APIENTRY
#   define GLUT_APIENTRY_DEFINED
#   if (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED) || defined(__BORLANDC__) || defined(__LCC__)
#    define APIENTRY    __stdcall
#   else
#    define APIENTRY
#   endif
#  endif

#  ifndef CALLBACK
#   if (defined(_M_MRX000) || defined(_M_IX86) || defined(_M_ALPHA) || defined(_M_PPC)) && !defined(MIDL_PASS) || defined(__LCC__)
#    define CALLBACK __stdcall
#   else
#    define CALLBACK
#   endif
#  endif

#  if defined( __LCC__ )
#   undef WINGDIAPI
#   define WINGDIAPI __stdcall
#  else

#   ifndef WINGDIAPI
#    define GLUT_WINGDIAPI_DEFINED
#    define WINGDIAPI __declspec(dllimport)
#   endif
#  endif

#  ifndef _WCHAR_T_DEFINED
typedef unsigned short wchar_t;
#   define _WCHAR_T_DEFINED
#  endif
# endif


# if !defined(GLUT_BUILDING_LIB) && !defined(GLUT_NO_LIB_PRAGMA)
#pragma comment (lib, "winmm.lib")      

#  ifdef GLUT_USE_SGI_OPENGL
#pragma comment (lib, "opengl.lib")    
#pragma comment (lib, "glu.lib")       
#pragma comment (lib, "glut.lib")      
#  else
#pragma comment (lib, "opengl32.lib")  
#pragma comment (lib, "glu32.lib")     
#pragma comment (lib, "glut32.lib")    
#  endif
# endif


# ifndef GLUT_NO_WARNING_DISABLE
#pragma warning (disable:4244)  
#pragma warning (disable:4305)  
# endif




# if !defined(_MSC_VER) && !defined(__cdecl)

#  define __cdecl
#  define GLUT_DEFINED___CDECL
# endif
# ifndef _CRTIMP
#  ifdef _NTSDK

#   define _CRTIMP
#  else

#   ifdef _DLL
#    define _CRTIMP __declspec(dllimport)
#   else
#    define _CRTIMP
#   endif
#  endif
#  define GLUT_DEFINED__CRTIMP
# endif


# ifdef GLUT_BUILDING_LIB
#  define GLUTAPI __declspec(dllexport)
# else
#  ifdef _DLL
#   define GLUTAPI __declspec(dllimport)
#  else
#   define GLUTAPI extern
#  endif
# endif


# define GLUTCALLBACK __cdecl

#endif  

#include <GL/gl.h>
#include <GL/glu.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
# ifndef GLUT_BUILDING_LIB
extern _CRTIMP void __cdecl exit(int);
# endif
#else


# define APIENTRY
# define GLUT_APIENTRY_DEFINED
# define CALLBACK

# define GLUTAPI extern
# define GLUTCALLBACK

extern void exit(int);
#endif


#ifndef GLUT_API_VERSION  
#define GLUT_API_VERSION		3
#endif


#ifndef GLUT_XLIB_IMPLEMENTATION  
#define GLUT_XLIB_IMPLEMENTATION	15
#endif


#define GLUT_RGB			0
#define GLUT_RGBA			GLUT_RGB
#define GLUT_INDEX			1
#define GLUT_SINGLE			0
#define GLUT_DOUBLE			2
#define GLUT_ACCUM			4
#define GLUT_ALPHA			8
#define GLUT_DEPTH			16
#define GLUT_STENCIL			32
#if (GLUT_API_VERSION >= 2)
#define GLUT_MULTISAMPLE		128
#define GLUT_STEREO			256
#endif
#if (GLUT_API_VERSION >= 3)
#define GLUT_LUMINANCE			512
#endif


#define GLUT_LEFT_BUTTON		0
#define GLUT_MIDDLE_BUTTON		1
#define GLUT_RIGHT_BUTTON		2


#define GLUT_DOWN			0
#define GLUT_UP				1

#if (GLUT_API_VERSION >= 2)

#define GLUT_KEY_F1			1
#define GLUT_KEY_F2			2
#define GLUT_KEY_F3			3
#define GLUT_KEY_F4			4
#define GLUT_KEY_F5			5
#define GLUT_KEY_F6			6
#define GLUT_KEY_F7			7
#define GLUT_KEY_F8			8
#define GLUT_KEY_F9			9
#define GLUT_KEY_F10			10
#define GLUT_KEY_F11			11
#define GLUT_KEY_F12			12

#define GLUT_KEY_LEFT			100
#define GLUT_KEY_UP			101
#define GLUT_KEY_RIGHT			102
#define GLUT_KEY_DOWN			103
#define GLUT_KEY_PAGE_UP		104
#define GLUT_KEY_PAGE_DOWN		105
#define GLUT_KEY_HOME			106
#define GLUT_KEY_END			107
#define GLUT_KEY_INSERT			108
#endif


#define GLUT_LEFT			0
#define GLUT_ENTERED			1


#define GLUT_MENU_NOT_IN_USE		0
#define GLUT_MENU_IN_USE		1


#define GLUT_NOT_VISIBLE		0
#define GLUT_VISIBLE			1


#define GLUT_HIDDEN			0
#define GLUT_FULLY_RETAINED		1
#define GLUT_PARTIALLY_RETAINED		2
#define GLUT_FULLY_COVERED		3


#define GLUT_RED			0
#define GLUT_GREEN			1
#define GLUT_BLUE			2

#if defined(_WIN32)

#define GLUT_STROKE_ROMAN		((void*)0)
#define GLUT_STROKE_MONO_ROMAN		((void*)1)


#define GLUT_BITMAP_9_BY_15		((void*)2)
#define GLUT_BITMAP_8_BY_13		((void*)3)
#define GLUT_BITMAP_TIMES_ROMAN_10	((void*)4)
#define GLUT_BITMAP_TIMES_ROMAN_24	((void*)5)
#if (GLUT_API_VERSION >= 3)
#define GLUT_BITMAP_HELVETICA_10	((void*)6)
#define GLUT_BITMAP_HELVETICA_12	((void*)7)
#define GLUT_BITMAP_HELVETICA_18	((void*)8)
#endif
#else

GLUTAPI void *glutStrokeRoman;
GLUTAPI void *glutStrokeMonoRoman;


#define GLUT_STROKE_ROMAN		(&glutStrokeRoman)
#define GLUT_STROKE_MONO_ROMAN		(&glutStrokeMonoRoman)


GLUTAPI void *glutBitmap9By15;
GLUTAPI void *glutBitmap8By13;
GLUTAPI void *glutBitmapTimesRoman10;
GLUTAPI void *glutBitmapTimesRoman24;
GLUTAPI void *glutBitmapHelvetica10;
GLUTAPI void *glutBitmapHelvetica12;
GLUTAPI void *glutBitmapHelvetica18;


#define GLUT_BITMAP_9_BY_15		(&glutBitmap9By15)
#define GLUT_BITMAP_8_BY_13		(&glutBitmap8By13)
#define GLUT_BITMAP_TIMES_ROMAN_10	(&glutBitmapTimesRoman10)
#define GLUT_BITMAP_TIMES_ROMAN_24	(&glutBitmapTimesRoman24)
#if (GLUT_API_VERSION >= 3)
#define GLUT_BITMAP_HELVETICA_10	(&glutBitmapHelvetica10)
#define GLUT_BITMAP_HELVETICA_12	(&glutBitmapHelvetica12)
#define GLUT_BITMAP_HELVETICA_18	(&glutBitmapHelvetica18)
#endif
#endif


#define GLUT_WINDOW_X			((GLenum) 100)
#define GLUT_WINDOW_Y			((GLenum) 101)
#define GLUT_WINDOW_WIDTH		((GLenum) 102)
#define GLUT_WINDOW_HEIGHT		((GLenum) 103)
#define GLUT_WINDOW_BUFFER_SIZE		((GLenum) 104)
#define GLUT_WINDOW_STENCIL_SIZE	((GLenum) 105)
#define GLUT_WINDOW_DEPTH_SIZE		((GLenum) 106)
#define GLUT_WINDOW_RED_SIZE		((GLenum) 107)
#define GLUT_WINDOW_GREEN_SIZE		((GLenum) 108)
#define GLUT_WINDOW_BLUE_SIZE		((GLenum) 109)
#define GLUT_WINDOW_ALPHA_SIZE		((GLenum) 110)
#define GLUT_WINDOW_ACCUM_RED_SIZE	((GLenum) 111)
#define GLUT_WINDOW_ACCUM_GREEN_SIZE	((GLenum) 112)
#define GLUT_WINDOW_ACCUM_BLUE_SIZE	((GLenum) 113)
#define GLUT_WINDOW_ACCUM_ALPHA_SIZE	((GLenum) 114)
#define GLUT_WINDOW_DOUBLEBUFFER	((GLenum) 115)
#define GLUT_WINDOW_RGBA		((GLenum) 116)
#define GLUT_WINDOW_PARENT		((GLenum) 117)
#define GLUT_WINDOW_NUM_CHILDREN	((GLenum) 118)
#define GLUT_WINDOW_COLORMAP_SIZE	((GLenum) 119)
#if (GLUT_API_VERSION >= 2)
#define GLUT_WINDOW_NUM_SAMPLES		((GLenum) 120)
#define GLUT_WINDOW_STEREO		((GLenum) 121)
#endif
#if (GLUT_API_VERSION >= 3)
#define GLUT_WINDOW_CURSOR		((GLenum) 122)
#endif
#define GLUT_SCREEN_WIDTH		((GLenum) 200)
#define GLUT_SCREEN_HEIGHT		((GLenum) 201)
#define GLUT_SCREEN_WIDTH_MM		((GLenum) 202)
#define GLUT_SCREEN_HEIGHT_MM		((GLenum) 203)
#define GLUT_MENU_NUM_ITEMS		((GLenum) 300)
#define GLUT_DISPLAY_MODE_POSSIBLE	((GLenum) 400)
#define GLUT_INIT_WINDOW_X		((GLenum) 500)
#define GLUT_INIT_WINDOW_Y		((GLenum) 501)
#define GLUT_INIT_WINDOW_WIDTH		((GLenum) 502)
#define GLUT_INIT_WINDOW_HEIGHT		((GLenum) 503)
#define GLUT_INIT_DISPLAY_MODE		((GLenum) 504)
#if (GLUT_API_VERSION >= 2)
#define GLUT_ELAPSED_TIME		((GLenum) 700)
#endif
#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 13)
#define GLUT_WINDOW_FORMAT_ID		((GLenum) 123)
#endif

#if (GLUT_API_VERSION >= 2)

#define GLUT_HAS_KEYBOARD		((GLenum) 600)
#define GLUT_HAS_MOUSE			((GLenum) 601)
#define GLUT_HAS_SPACEBALL		((GLenum) 602)
#define GLUT_HAS_DIAL_AND_BUTTON_BOX	((GLenum) 603)
#define GLUT_HAS_TABLET			((GLenum) 604)
#define GLUT_NUM_MOUSE_BUTTONS		((GLenum) 605)
#define GLUT_NUM_SPACEBALL_BUTTONS	((GLenum) 606)
#define GLUT_NUM_BUTTON_BOX_BUTTONS	((GLenum) 607)
#define GLUT_NUM_DIALS			((GLenum) 608)
#define GLUT_NUM_TABLET_BUTTONS		((GLenum) 609)
#endif
#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 13)
#define GLUT_DEVICE_IGNORE_KEY_REPEAT   ((GLenum) 610)
#define GLUT_DEVICE_KEY_REPEAT          ((GLenum) 611)
#define GLUT_HAS_JOYSTICK		((GLenum) 612)
#define GLUT_OWNS_JOYSTICK		((GLenum) 613)
#define GLUT_JOYSTICK_BUTTONS		((GLenum) 614)
#define GLUT_JOYSTICK_AXES		((GLenum) 615)
#define GLUT_JOYSTICK_POLL_RATE		((GLenum) 616)
#endif

#if (GLUT_API_VERSION >= 3)

#define GLUT_OVERLAY_POSSIBLE           ((GLenum) 800)
#define GLUT_LAYER_IN_USE		((GLenum) 801)
#define GLUT_HAS_OVERLAY		((GLenum) 802)
#define GLUT_TRANSPARENT_INDEX		((GLenum) 803)
#define GLUT_NORMAL_DAMAGED		((GLenum) 804)
#define GLUT_OVERLAY_DAMAGED		((GLenum) 805)

#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 9)

#define GLUT_VIDEO_RESIZE_POSSIBLE	((GLenum) 900)
#define GLUT_VIDEO_RESIZE_IN_USE	((GLenum) 901)
#define GLUT_VIDEO_RESIZE_X_DELTA	((GLenum) 902)
#define GLUT_VIDEO_RESIZE_Y_DELTA	((GLenum) 903)
#define GLUT_VIDEO_RESIZE_WIDTH_DELTA	((GLenum) 904)
#define GLUT_VIDEO_RESIZE_HEIGHT_DELTA	((GLenum) 905)
#define GLUT_VIDEO_RESIZE_X		((GLenum) 906)
#define GLUT_VIDEO_RESIZE_Y		((GLenum) 907)
#define GLUT_VIDEO_RESIZE_WIDTH		((GLenum) 908)
#define GLUT_VIDEO_RESIZE_HEIGHT	((GLenum) 909)
#endif


#define GLUT_NORMAL			((GLenum) 0)
#define GLUT_OVERLAY			((GLenum) 1)


#define GLUT_ACTIVE_SHIFT               1
#define GLUT_ACTIVE_CTRL                2
#define GLUT_ACTIVE_ALT                 4



#define GLUT_CURSOR_RIGHT_ARROW		0
#define GLUT_CURSOR_LEFT_ARROW		1

#define GLUT_CURSOR_INFO		2
#define GLUT_CURSOR_DESTROY		3
#define GLUT_CURSOR_HELP		4
#define GLUT_CURSOR_CYCLE		5
#define GLUT_CURSOR_SPRAY		6
#define GLUT_CURSOR_WAIT		7
#define GLUT_CURSOR_TEXT		8
#define GLUT_CURSOR_CROSSHAIR		9

#define GLUT_CURSOR_UP_DOWN		10
#define GLUT_CURSOR_LEFT_RIGHT		11

#define GLUT_CURSOR_TOP_SIDE		12
#define GLUT_CURSOR_BOTTOM_SIDE		13
#define GLUT_CURSOR_LEFT_SIDE		14
#define GLUT_CURSOR_RIGHT_SIDE		15
#define GLUT_CURSOR_TOP_LEFT_CORNER	16
#define GLUT_CURSOR_TOP_RIGHT_CORNER	17
#define GLUT_CURSOR_BOTTOM_RIGHT_CORNER	18
#define GLUT_CURSOR_BOTTOM_LEFT_CORNER	19

#define GLUT_CURSOR_INHERIT		100

#define GLUT_CURSOR_NONE		101

#define GLUT_CURSOR_FULL_CROSSHAIR	102
#endif


GLUTAPI void APIENTRY glutInit(int *argcp, char **argv);
#if defined(_WIN32) && !defined(GLUT_DISABLE_ATEXIT_HACK)
GLUTAPI void APIENTRY __glutInitWithExit(int *argcp, char **argv, void (__cdecl *exitfunc)(int));
#ifndef GLUT_BUILDING_LIB
static void APIENTRY glutInit_ATEXIT_HACK(int *argcp, char **argv) { __glutInitWithExit(argcp, argv, exit); }
#define glutInit glutInit_ATEXIT_HACK
#endif
#endif
GLUTAPI void APIENTRY glutInitDisplayMode(unsigned int mode);
#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 9)
GLUTAPI void APIENTRY glutInitDisplayString(const char *string);
#endif
GLUTAPI void APIENTRY glutInitWindowPosition(int x, int y);
GLUTAPI void APIENTRY glutInitWindowSize(int width, int height);
GLUTAPI void APIENTRY glutMainLoop(void);


GLUTAPI int APIENTRY glutCreateWindow(const char *title);
#if defined(_WIN32) && !defined(GLUT_DISABLE_ATEXIT_HACK)
GLUTAPI int APIENTRY __glutCreateWindowWithExit(const char *title, void (__cdecl *exitfunc)(int));
#ifndef GLUT_BUILDING_LIB
static int APIENTRY glutCreateWindow_ATEXIT_HACK(const char *title) { return __glutCreateWindowWithExit(title, exit); }
#define glutCreateWindow glutCreateWindow_ATEXIT_HACK
#endif
#endif
GLUTAPI int APIENTRY glutCreateSubWindow(int win, int x, int y, int width, int height);
GLUTAPI void APIENTRY glutDestroyWindow(int win);
GLUTAPI void APIENTRY glutPostRedisplay(void);
#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 11)
GLUTAPI void APIENTRY glutPostWindowRedisplay(int win);
#endif
GLUTAPI void APIENTRY glutSwapBuffers(void);
GLUTAPI int APIENTRY glutGetWindow(void);
GLUTAPI void APIENTRY glutSetWindow(int win);
GLUTAPI void APIENTRY glutSetWindowTitle(const char *title);
GLUTAPI void APIENTRY glutSetIconTitle(const char *title);
GLUTAPI void APIENTRY glutPositionWindow(int x, int y);
GLUTAPI void APIENTRY glutReshapeWindow(int width, int height);
GLUTAPI void APIENTRY glutPopWindow(void);
GLUTAPI void APIENTRY glutPushWindow(void);
GLUTAPI void APIENTRY glutIconifyWindow(void);
GLUTAPI void APIENTRY glutShowWindow(void);
GLUTAPI void APIENTRY glutHideWindow(void);
#if (GLUT_API_VERSION >= 3)
GLUTAPI void APIENTRY glutFullScreen(void);
GLUTAPI void APIENTRY glutSetCursor(int cursor);
#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 9)
GLUTAPI void APIENTRY glutWarpPointer(int x, int y);
#endif


GLUTAPI void APIENTRY glutEstablishOverlay(void);
GLUTAPI void APIENTRY glutRemoveOverlay(void);
GLUTAPI void APIENTRY glutUseLayer(GLenum layer);
GLUTAPI void APIENTRY glutPostOverlayRedisplay(void);
#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 11)
GLUTAPI void APIENTRY glutPostWindowOverlayRedisplay(int win);
#endif
GLUTAPI void APIENTRY glutShowOverlay(void);
GLUTAPI void APIENTRY glutHideOverlay(void);
#endif


GLUTAPI int APIENTRY glutCreateMenu(void (GLUTCALLBACK *func)(int));
#if defined(_WIN32) && !defined(GLUT_DISABLE_ATEXIT_HACK)
GLUTAPI int APIENTRY __glutCreateMenuWithExit(void (GLUTCALLBACK *func)(int), void (__cdecl *exitfunc)(int));
#ifndef GLUT_BUILDING_LIB
static int APIENTRY glutCreateMenu_ATEXIT_HACK(void (GLUTCALLBACK *func)(int)) { return __glutCreateMenuWithExit(func, exit); }
#define glutCreateMenu glutCreateMenu_ATEXIT_HACK
#endif
#endif
GLUTAPI void APIENTRY glutDestroyMenu(int menu);
GLUTAPI int APIENTRY glutGetMenu(void);
GLUTAPI void APIENTRY glutSetMenu(int menu);
GLUTAPI void APIENTRY glutAddMenuEntry(const char *label, int value);
GLUTAPI void APIENTRY glutAddSubMenu(const char *label, int submenu);
GLUTAPI void APIENTRY glutChangeToMenuEntry(int item, const char *label, int value);
GLUTAPI void APIENTRY glutChangeToSubMenu(int item, const char *label, int submenu);
GLUTAPI void APIENTRY glutRemoveMenuItem(int item);
GLUTAPI void APIENTRY glutAttachMenu(int button);
GLUTAPI void APIENTRY glutDetachMenu(int button);


GLUTAPI void APIENTRY glutDisplayFunc(void (GLUTCALLBACK *func)(void));
GLUTAPI void APIENTRY glutReshapeFunc(void (GLUTCALLBACK *func)(int width, int height));
GLUTAPI void APIENTRY glutKeyboardFunc(void (GLUTCALLBACK *func)(unsigned char key, int x, int y));
GLUTAPI void APIENTRY glutMouseFunc(void (GLUTCALLBACK *func)(int button, int state, int x, int y));
GLUTAPI void APIENTRY glutMotionFunc(void (GLUTCALLBACK *func)(int x, int y));
GLUTAPI void APIENTRY glutPassiveMotionFunc(void (GLUTCALLBACK *func)(int x, int y));
GLUTAPI void APIENTRY glutEntryFunc(void (GLUTCALLBACK *func)(int state));
GLUTAPI void APIENTRY glutVisibilityFunc(void (GLUTCALLBACK *func)(int state));
GLUTAPI void APIENTRY glutIdleFunc(void (GLUTCALLBACK *func)(void));
GLUTAPI void APIENTRY glutTimerFunc(unsigned int millis, void (GLUTCALLBACK *func)(int value), int value);
GLUTAPI void APIENTRY glutMenuStateFunc(void (GLUTCALLBACK *func)(int state));
#if (GLUT_API_VERSION >= 2)
GLUTAPI void APIENTRY glutSpecialFunc(void (GLUTCALLBACK *func)(int key, int x, int y));
GLUTAPI void APIENTRY glutSpaceballMotionFunc(void (GLUTCALLBACK *func)(int x, int y, int z));
GLUTAPI void APIENTRY glutSpaceballRotateFunc(void (GLUTCALLBACK *func)(int x, int y, int z));
GLUTAPI void APIENTRY glutSpaceballButtonFunc(void (GLUTCALLBACK *func)(int button, int state));
GLUTAPI void APIENTRY glutButtonBoxFunc(void (GLUTCALLBACK *func)(int button, int state));
GLUTAPI void APIENTRY glutDialsFunc(void (GLUTCALLBACK *func)(int dial, int value));
GLUTAPI void APIENTRY glutTabletMotionFunc(void (GLUTCALLBACK *func)(int x, int y));
GLUTAPI void APIENTRY glutTabletButtonFunc(void (GLUTCALLBACK *func)(int button, int state, int x, int y));
#if (GLUT_API_VERSION >= 3)
GLUTAPI void APIENTRY glutMenuStatusFunc(void (GLUTCALLBACK *func)(int status, int x, int y));
GLUTAPI void APIENTRY glutOverlayDisplayFunc(void (GLUTCALLBACK *func)(void));
#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 9)
GLUTAPI void APIENTRY glutWindowStatusFunc(void (GLUTCALLBACK *func)(int state));
#endif
#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 13)
GLUTAPI void APIENTRY glutKeyboardUpFunc(void (GLUTCALLBACK *func)(unsigned char key, int x, int y));
GLUTAPI void APIENTRY glutSpecialUpFunc(void (GLUTCALLBACK *func)(int key, int x, int y));
GLUTAPI void APIENTRY glutJoystickFunc(void (GLUTCALLBACK *func)(unsigned int buttonMask, int x, int y, int z), int pollInterval);
#endif
#endif
#endif


GLUTAPI void APIENTRY glutSetColor(int, GLfloat red, GLfloat green, GLfloat blue);
GLUTAPI GLfloat APIENTRY glutGetColor(int ndx, int component);
GLUTAPI void APIENTRY glutCopyColormap(int win);


GLUTAPI int APIENTRY glutGet(GLenum type);
GLUTAPI int APIENTRY glutDeviceGet(GLenum type);
#if (GLUT_API_VERSION >= 2)

GLUTAPI int APIENTRY glutExtensionSupported(const char *name);
#endif
#if (GLUT_API_VERSION >= 3)
GLUTAPI int APIENTRY glutGetModifiers(void);
GLUTAPI int APIENTRY glutLayerGet(GLenum type);
#endif


GLUTAPI void APIENTRY glutBitmapCharacter(void *font, int character);
GLUTAPI int APIENTRY glutBitmapWidth(void *font, int character);
GLUTAPI void APIENTRY glutStrokeCharacter(void *font, int character);
GLUTAPI int APIENTRY glutStrokeWidth(void *font, int character);
#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 9)
GLUTAPI int APIENTRY glutBitmapLength(void *font, const unsigned char *string);
GLUTAPI int APIENTRY glutStrokeLength(void *font, const unsigned char *string);
#endif


GLUTAPI void APIENTRY glutWireSphere(GLdouble radius, GLint slices, GLint stacks);
GLUTAPI void APIENTRY glutSolidSphere(GLdouble radius, GLint slices, GLint stacks);
GLUTAPI void APIENTRY glutWireCone(GLdouble base, GLdouble height, GLint slices, GLint stacks);
GLUTAPI void APIENTRY glutSolidCone(GLdouble base, GLdouble height, GLint slices, GLint stacks);
GLUTAPI void APIENTRY glutWireCube(GLdouble size);
GLUTAPI void APIENTRY glutSolidCube(GLdouble size);
GLUTAPI void APIENTRY glutWireTorus(GLdouble innerRadius, GLdouble outerRadius, GLint sides, GLint rings);
GLUTAPI void APIENTRY glutSolidTorus(GLdouble innerRadius, GLdouble outerRadius, GLint sides, GLint rings);
GLUTAPI void APIENTRY glutWireDodecahedron(void);
GLUTAPI void APIENTRY glutSolidDodecahedron(void);
GLUTAPI void APIENTRY glutWireTeapot(GLdouble size);
GLUTAPI void APIENTRY glutSolidTeapot(GLdouble size);
GLUTAPI void APIENTRY glutWireOctahedron(void);
GLUTAPI void APIENTRY glutSolidOctahedron(void);
GLUTAPI void APIENTRY glutWireTetrahedron(void);
GLUTAPI void APIENTRY glutSolidTetrahedron(void);
GLUTAPI void APIENTRY glutWireIcosahedron(void);
GLUTAPI void APIENTRY glutSolidIcosahedron(void);

#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 9)

GLUTAPI int APIENTRY glutVideoResizeGet(GLenum param);
GLUTAPI void APIENTRY glutSetupVideoResizing(void);
GLUTAPI void APIENTRY glutStopVideoResizing(void);
GLUTAPI void APIENTRY glutVideoResize(int x, int y, int width, int height);
GLUTAPI void APIENTRY glutVideoPan(int x, int y, int width, int height);


GLUTAPI void APIENTRY glutReportErrors(void);
#endif

#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 13)


#define GLUT_KEY_REPEAT_OFF		0
#define GLUT_KEY_REPEAT_ON		1
#define GLUT_KEY_REPEAT_DEFAULT		2


#define GLUT_JOYSTICK_BUTTON_A		1
#define GLUT_JOYSTICK_BUTTON_B		2
#define GLUT_JOYSTICK_BUTTON_C		4
#define GLUT_JOYSTICK_BUTTON_D		8

GLUTAPI void APIENTRY glutIgnoreKeyRepeat(int ignore);
GLUTAPI void APIENTRY glutSetKeyRepeat(int repeatMode);
GLUTAPI void APIENTRY glutForceJoystickFunc(void);



#define GLUT_GAME_MODE_ACTIVE           ((GLenum) 0)
#define GLUT_GAME_MODE_POSSIBLE         ((GLenum) 1)
#define GLUT_GAME_MODE_WIDTH            ((GLenum) 2)
#define GLUT_GAME_MODE_HEIGHT           ((GLenum) 3)
#define GLUT_GAME_MODE_PIXEL_DEPTH      ((GLenum) 4)
#define GLUT_GAME_MODE_REFRESH_RATE     ((GLenum) 5)
#define GLUT_GAME_MODE_DISPLAY_CHANGED  ((GLenum) 6)

GLUTAPI void APIENTRY glutGameModeString(const char *string);
GLUTAPI int APIENTRY glutEnterGameMode(void);
GLUTAPI void APIENTRY glutLeaveGameMode(void);
GLUTAPI int APIENTRY glutGameModeGet(GLenum mode);
#endif

#ifdef __cplusplus
}

#endif

#ifdef GLUT_APIENTRY_DEFINED
# undef GLUT_APIENTRY_DEFINED
# undef APIENTRY
#endif

#ifdef GLUT_WINGDIAPI_DEFINED
# undef GLUT_WINGDIAPI_DEFINED
# undef WINGDIAPI
#endif

#ifdef GLUT_DEFINED___CDECL
# undef GLUT_DEFINED___CDECL
# undef __cdecl
#endif

#ifdef GLUT_DEFINED__CRTIMP
# undef GLUT_DEFINED__CRTIMP
# undef _CRTIMP
#endif

#endif                  
