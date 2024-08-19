#ifdef __vxworks
#include "vxWorks.h"
#endif
#ifdef IN_RTS
#include "tconfig.h"
#include "tsystem.h"
#define xmalloc(S) malloc (S)
#define xrealloc(V,S) realloc (V,S)
#else
#include "config.h"
#include "system.h"
#endif
#include "raise.h"
#include <fcntl.h>
#ifdef __cplusplus
extern "C" {
#endif
extern void __gnat_install_handler(void);
int __gnat_wide_text_translation_required = 0;
int __gnat_rt_init_count = 0;
#if defined (__MINGW32__)
#include "mingw32.h"
#include <windows.h>
extern void __gnat_init_float (void);
extern int gnat_argc;
extern char **gnat_argv;
extern CRITICAL_SECTION ProcListCS;
extern HANDLE ProcListEvt;
#ifdef GNAT_UNICODE_SUPPORT
#define EXPAND_ARGV_RATE 128
int __gnat_do_argv_expansion = 1;
#pragma weak __gnat_do_argv_expansion
static void
append_arg (int *index, LPWSTR dir, LPWSTR value,
char ***argv, int *last, int quoted)
{
int size;
LPWSTR fullvalue;
int vallen = _tcslen (value);
int dirlen;
if (dir == NULL)
{
dirlen = 0;
fullvalue = (LPWSTR) xmalloc ((vallen + 1) * sizeof(TCHAR));
}
else
{
dirlen = _tcslen (dir);
fullvalue = (LPWSTR) xmalloc ((dirlen + vallen + 1) * sizeof(TCHAR));
_tcscpy (fullvalue, dir);
}
if (quoted)
{
_tcsncpy (fullvalue + dirlen, value + 1, vallen - 1);
fullvalue [dirlen + vallen - sizeof(TCHAR)] = _T('\0');
}
else
_tcscpy (fullvalue + dirlen, value);
if (*last <= *index)
{
*last += EXPAND_ARGV_RATE;
*argv = (char **) xrealloc (*argv, (*last) * sizeof (char *));
}
size = WS2SC (NULL, fullvalue, 0);
(*argv)[*index] = (char *) xmalloc (size + sizeof(TCHAR));
WS2SC ((*argv)[*index], fullvalue, size);
free (fullvalue);
(*index)++;
}
#endif
void
__gnat_runtime_initialize(int install_handler)
{
__gnat_rt_init_count++;
if (__gnat_rt_init_count > 1)
return;
__gnat_init_float ();
InitializeCriticalSection (&ProcListCS);
ProcListEvt = CreateEvent (NULL, FALSE, FALSE, NULL);
#ifdef GNAT_UNICODE_SUPPORT
{
char *codepage = getenv ("GNAT_CODE_PAGE");
__gnat_current_codepage = CP_UTF8;
if (codepage != NULL)
{
if (strcmp (codepage, "CP_ACP") == 0)
__gnat_current_codepage = CP_ACP;
else if (strcmp (codepage, "CP_UTF8") == 0)
__gnat_current_codepage = CP_UTF8;
}
}
{
char *ccsencoding = getenv ("GNAT_CCS_ENCODING");
__gnat_current_ccs_encoding = _O_TEXT;
__gnat_wide_text_translation_required = 0;
if (ccsencoding != NULL)
{
if (strcmp (ccsencoding, "U16TEXT") == 0)
{
__gnat_current_ccs_encoding = _O_U16TEXT;
__gnat_wide_text_translation_required = 1;
}
else if (strcmp (ccsencoding, "TEXT") == 0)
{
__gnat_current_ccs_encoding = _O_TEXT;
__gnat_wide_text_translation_required = 0;
}
else if (strcmp (ccsencoding, "WTEXT") == 0)
{
__gnat_current_ccs_encoding = _O_WTEXT;
__gnat_wide_text_translation_required = 1;
}
else if (strcmp (ccsencoding, "U8TEXT") == 0)
{
__gnat_current_ccs_encoding = _O_U8TEXT;
__gnat_wide_text_translation_required = 1;
}
}
}
{
LPWSTR *wargv;
int wargc;
int k;
int last;
int argc_expanded = 0;
TCHAR result [MAX_PATH];
int quoted;
wargv = CommandLineToArgvW (GetCommandLineW(), &wargc);
if (wargv != NULL)
{
last = wargc + 1;
gnat_argv = (char **) xmalloc ((last) * sizeof (char *));
SearchPath (NULL, wargv[0], _T(".exe"), MAX_PATH, result, NULL);
append_arg (&argc_expanded, NULL, result, &gnat_argv, &last, 0);
for (k=1; k<wargc; k++)
{
quoted = (wargv[k][0] == _T('\''));
if (!quoted && __gnat_do_argv_expansion
&& (_tcsstr (wargv[k], _T("?")) != 0 ||
_tcsstr (wargv[k], _T("*")) != 0))
{
WIN32_FIND_DATA FileData;
HANDLE hDir = FindFirstFile (wargv[k], &FileData);
LPWSTR dir = NULL;
LPWSTR ldir = _tcsrchr (wargv[k], _T('\\'));
if (ldir == NULL)
ldir = _tcsrchr (wargv[k], _T('/'));
if (hDir == INVALID_HANDLE_VALUE)
{
append_arg (&argc_expanded, NULL, wargv[k],
&gnat_argv, &last, quoted);
}
else
{
if (ldir != NULL)
{
int n = ldir - wargv[k] + 1;
dir = (LPWSTR) xmalloc ((n + 1) * sizeof (TCHAR));
_tcsncpy (dir, wargv[k], n);
dir[n] = _T('\0');
}
do {
if (_tcscmp (FileData.cFileName, _T(".")) != 0
&& _tcscmp (FileData.cFileName, _T("..")) != 0)
append_arg (&argc_expanded, dir, FileData.cFileName,
&gnat_argv, &last, 0);
} while (FindNextFile (hDir, &FileData));
FindClose (hDir);
if (dir != NULL)
free (dir);
}
}
else
{
append_arg (&argc_expanded, NULL, wargv[k],
&gnat_argv, &last,
quoted && __gnat_do_argv_expansion);
}
}
LocalFree (wargv);
gnat_argc = argc_expanded;
gnat_argv = (char **) xrealloc
(gnat_argv, argc_expanded * sizeof (char *));
}
}
#endif
if (install_handler)
__gnat_install_handler();
}
#elif defined (__Lynx__) || defined (__FreeBSD__) || defined(__NetBSD__) \
|| defined (__OpenBSD__)
extern void __gnat_init_float (void);
void
__gnat_runtime_initialize(int install_handler)
{
__gnat_rt_init_count++;
if (__gnat_rt_init_count > 1)
return;
__gnat_init_float ();
if (install_handler)
__gnat_install_handler();
}
#elif defined(__vxworks)
extern void __gnat_init_float (void);
void
__gnat_runtime_initialize(int install_handler)
{
__gnat_rt_init_count++;
if (__gnat_rt_init_count > 1)
return;
__gnat_init_float ();
if (install_handler)
__gnat_install_handler();
}
#else
void
__gnat_runtime_initialize(int install_handler)
{
__gnat_rt_init_count++;
if (__gnat_rt_init_count > 1)
return;
if (install_handler)
__gnat_install_handler();
}
#endif
#ifdef __cplusplus
}
#endif
