#include <windows.h>
#include <tchar.h>
#include <dbghelp.h>
#include <process.h>
#include <stdio.h>
#pragma comment ( lib, "dbghelp.lib" )
typedef unsigned (__stdcall *PTHREAD_START) (void*);
#define BEGINTHREADEX(lpsa, cbStack, lpStartAddr,\
lpvThreadParm, fdwCreate, lpIDThread)          \
((HANDLE)_beginthreadex(                     \
(void*)(lpsa),                           \
(unsigned)(cbStack),                     \
(PTHREAD_START)(lpStartAddr),            \
(void*)(lpvThreadParm),                  \
(unsigned)(fdwCreate),                   \
(unsigned*)(lpIDThread)))
DWORD WINAPI WorkerThread( LPVOID lpParam );
void TestFunc( int* pParam );
LONG __stdcall MyCustomFilter( EXCEPTION_POINTERS* pep ); 
void CreateMiniDump( EXCEPTION_POINTERS* pep ); 
int _tmain( int argc, TCHAR* argv[] )
{
SetUnhandledExceptionFilter( MyCustomFilter ); 
_tprintf( _T("Starting the worker thread...\n") );
HANDLE  hThread   = NULL;
DWORD   ThreadId  = 0;
hThread = BEGINTHREADEX(0, 0, WorkerThread, 0, 0, &ThreadId );
if( hThread == NULL )
{
_tprintf( _T("Cannot start thread. Error: %u\n"), GetLastError() );
return 0;
}
_tprintf( _T("Worker thread started.\n") );
Sleep( 60 * 60 * 1000 );
_tprintf( _T("Test complete.\n") );
return 0;
}
DWORD WINAPI WorkerThread( LPVOID lpParam )
{
_tprintf( _T("Worker thread [%u] started.\n"), GetCurrentThreadId() );
Sleep( 10 * 1000 );
int* TempPtr = (int*)lpParam;
TestFunc( TempPtr );
return 0;
}
void TestFunc( int* pParam )
{
_tprintf( _T("TestFunc()\n") );
*pParam = 0;
}
LONG __stdcall MyCustomFilter( EXCEPTION_POINTERS* pep ) 
{
CreateMiniDump( pep ); 
return EXCEPTION_EXECUTE_HANDLER; 
}
void CreateMiniDump( EXCEPTION_POINTERS* pep ) 
{
HANDLE hFile = CreateFile( _T("MiniDump.dmp"), GENERIC_READ | GENERIC_WRITE, 
0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL ); 
if( ( hFile != NULL ) && ( hFile != INVALID_HANDLE_VALUE ) ) 
{
MINIDUMP_EXCEPTION_INFORMATION mdei; 
mdei.ThreadId           = GetCurrentThreadId(); 
mdei.ExceptionPointers  = pep; 
mdei.ClientPointers     = FALSE; 
MINIDUMP_TYPE mdt       = MiniDumpWithThreadInfo;                                   
BOOL rv = MiniDumpWriteDump( GetCurrentProcess(), GetCurrentProcessId(), 
hFile, mdt, (pep != 0) ? &mdei : 0, 0, 0 ); 
if( !rv ) 
_tprintf( _T("MiniDumpWriteDump failed. Error: %u \n"), GetLastError() ); 
else 
_tprintf( _T("Minidump created.\n") ); 
CloseHandle( hFile ); 
}
else 
{
_tprintf( _T("CreateFile failed. Error: %u \n"), GetLastError() ); 
}
}
