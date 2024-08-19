





#include "jam.h"
#include "output.h"
#ifdef USE_EXECNT
#include "execcmd.h"

#include "lists.h"
#include "output.h"
#include "pathsys.h"
#include "string.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#include <tlhelp32.h>

#if defined(__GNUC__) || defined(__clang__)
#else
#pragma warning( push )
#pragma warning(disable: 4800) 
#endif
#include <versionhelpers.h>
#if defined(__GNUC__) || defined(__clang__)
#else
#pragma warning( pop )
#endif



static int maxline();

static long raw_command_length( char const * command );

static FILETIME add_64(
unsigned long h1, unsigned long l1,
unsigned long h2, unsigned long l2 );

static FILETIME add_FILETIME( FILETIME t1, FILETIME t2 );

static FILETIME negate_FILETIME( FILETIME t );

static void record_times( HANDLE const, timing_info * const );

static double running_time( HANDLE const );

static void kill_process_tree( DWORD const procesdId, HANDLE const );

static int try_wait( int const timeoutMillis );

static void read_output();

static int try_kill_one();

static int is_parent_child( DWORD const parent, DWORD const child );

static void close_alert( PROCESS_INFORMATION const * const );

static void close_alerts();

static char const * prepare_command_file( string const * command, int slot );

static void invoke_cmd( char const * const command, int const slot );

static int get_free_cmdtab_slot();

static void string_new_from_argv( string * result, char const * const * argv );

static void string_renew( string * const );

static void reportWindowsError( char const * const apiName, int slot );

static void closeWinHandle( HANDLE * const handle );

static void register_wait( int job_id );




#define MAX_RAW_COMMAND_LENGTH 32766


#define IO_BUFFER_SIZE ( 64 * 1024 )


#define EXECCMD_PIPE_READ 0
#define EXECCMD_PIPE_WRITE 1

static int intr_installed;



static struct _cmdtab_t
{

string command_file[ 1 ];


HANDLE pipe_out[ 2 ];
HANDLE pipe_err[ 2 ];

string buffer_out[ 1 ];  
string buffer_err[ 1 ];  

PROCESS_INFORMATION pi;  

HANDLE wait_handle;

int flags;


ExecCmdCallback func;


void * closure;
} * cmdtab = NULL;
static int cmdtab_size = 0;


struct
{
int job_index;
HANDLE read_okay;
HANDLE write_okay;
} process_queue;



void execnt_unit_test()
{
#if !defined( NDEBUG )

{
typedef struct test { const char * command; int result; } test;
test tests[] = {
{ "", 0 },
{ "  ", 0 },
{ "x", 1 },
{ "\nx", 1 },
{ "x\n", 1 },
{ "\nx\n", 1 },
{ "\nx \n", 2 },
{ "\nx \n ", 2 },
{ " \n\t\t\v\r\r\n \t  x  \v \t\t\r\n\n\n   \n\n\v\t", 8 },
{ "x\ny", -1 },
{ "x\n\n y", -1 },
{ "echo x > foo.bar", -1 },
{ "echo x < foo.bar", -1 },
{ "echo x | foo.bar", -1 },
{ "echo x \">\" foo.bar", 18 },
{ "echo x '<' foo.bar", 18 },
{ "echo x \"|\" foo.bar", 18 },
{ "echo x \\\">\\\" foo.bar", -1 },
{ "echo x \\\"<\\\" foo.bar", -1 },
{ "echo x \\\"|\\\" foo.bar", -1 },
{ "\"echo x > foo.bar\"", 18 },
{ "echo x \"'\"<' foo.bar", -1 },
{ "echo x \\\\\"<\\\\\" foo.bar", 22 },
{ "echo x \\x\\\"<\\\\\" foo.bar", -1 },
{ 0 } };
test const * t;
for ( t = tests; t->command; ++t )
assert( raw_command_length( t->command ) == t->result );
}

{
int const length = maxline() + 9;
char * const cmd = (char *)BJAM_MALLOC_ATOMIC( length + 1 );
memset( cmd, 'x', length );
cmd[ length ] = 0;
assert( raw_command_length( cmd ) == length );
BJAM_FREE( cmd );
}
#endif
}


void exec_init( void )
{
if ( globs.jobs > cmdtab_size )
{
cmdtab = (_cmdtab_t*)BJAM_REALLOC( cmdtab, globs.jobs * sizeof( *cmdtab ) );
memset( cmdtab + cmdtab_size, 0, ( globs.jobs - cmdtab_size ) * sizeof( *cmdtab ) );
cmdtab_size = globs.jobs;
}
if ( globs.jobs > MAXIMUM_WAIT_OBJECTS && !process_queue.read_okay )
{
process_queue.read_okay = CreateEvent( NULL, FALSE, FALSE, NULL );
process_queue.write_okay = CreateEvent( NULL, FALSE, TRUE, NULL );
}
}


void exec_done( void )
{
if ( process_queue.read_okay )
{
CloseHandle( process_queue.read_okay );
}
if ( process_queue.write_okay )
{
CloseHandle( process_queue.write_okay );
}
BJAM_FREE( cmdtab );
}



int exec_check
(
string const * command,
LIST * * pShell,
int * error_length,
int * error_max_length
)
{

if ( list_empty( *pShell ) )
{
char const * s = command->value;
while ( isspace( *s ) ) ++s;
if ( !*s )
return EXEC_CHECK_NOOP;
}


if ( is_raw_command_request( *pShell ) )
{
long const raw_cmd_length = raw_command_length( command->value );
if ( raw_cmd_length < 0 )
{

list_free( *pShell );
*pShell = L0;
}
else if ( raw_cmd_length > MAX_RAW_COMMAND_LENGTH )
{
*error_length = raw_cmd_length;
*error_max_length = MAX_RAW_COMMAND_LENGTH;
return EXEC_CHECK_TOO_LONG;
}
else
return raw_cmd_length ? EXEC_CHECK_OK : EXEC_CHECK_NOOP;
}




return check_cmd_for_too_long_lines( command->value, maxline(),
error_length, error_max_length );
}




void exec_cmd
(
string const * cmd_orig,
int flags,
ExecCmdCallback func,
void * closure,
LIST * shell
)
{
int const slot = get_free_cmdtab_slot();
int const is_raw_cmd = is_raw_command_request( shell );
string cmd_local[ 1 ];


static LIST * default_shell;
if ( !default_shell )
default_shell = list_new( object_new( "cmd.exe /Q/C" ) );


if ( list_empty( shell ) )
shell = default_shell;

if ( DEBUG_EXECCMD )
{
if ( is_raw_cmd )
out_printf( "Executing raw command directly\n" );
else
{
out_printf( "Executing using a command file and the shell: " );
list_print( shell );
out_printf( "\n" );
}
}


if ( is_raw_cmd )
{
char const * start = cmd_orig->value;
char const * p = cmd_orig->value + cmd_orig->size;
char const * end = p;
while ( isspace( *start ) ) ++start;
while ( p > start && isspace( p[ -1 ] ) )
if ( *--p == '\n' )
end = p;
string_new( cmd_local );
string_append_range( cmd_local, start, end );
assert( long(cmd_local->size) == raw_command_length( cmd_orig->value ) );
}

else
{
char const * const cmd_file = prepare_command_file( cmd_orig, slot );
char const * argv[ MAXARGC + 1 ];  
argv_from_shell( argv, shell, cmd_file, slot );
string_new_from_argv( cmd_local, argv );
}


if ( !intr_installed )
{
intr_installed = 1;
signal( SIGINT, onintr );
}

cmdtab[ slot ].flags = flags;


cmdtab[ slot ].func = func;
cmdtab[ slot ].closure = closure;


invoke_cmd( cmd_local->value, slot );


string_free( cmd_local );
}




void exec_wait()
{
int i = -1;
int exit_reason;  


while ( 1 )
{

i = try_wait( 500 );

read_output();

close_alerts();

if ( i >= 0 ) { exit_reason = EXIT_OK; break; }

i = try_kill_one();
if ( i >= 0 ) { exit_reason = EXIT_TIMEOUT; break; }
}


{
DWORD exit_code;
timing_info time;
int rstat;


record_times( cmdtab[ i ].pi.hProcess, &time );


if ( cmdtab[ i ].command_file->size )
unlink( cmdtab[ i ].command_file->value );


GetExitCodeProcess( cmdtab[ i ].pi.hProcess, &exit_code );


if ( interrupted() )
rstat = EXEC_CMD_INTR;
else if ( exit_code )
rstat = EXEC_CMD_FAIL;
else
rstat = EXEC_CMD_OK;


(*cmdtab[ i ].func)( cmdtab[ i ].closure, rstat, &time,
cmdtab[ i ].buffer_out->value, cmdtab[ i ].buffer_err->value,
exit_reason );


closeWinHandle( &cmdtab[ i ].pi.hProcess );
closeWinHandle( &cmdtab[ i ].pi.hThread );
closeWinHandle( &cmdtab[ i ].pipe_out[ EXECCMD_PIPE_READ ] );
closeWinHandle( &cmdtab[ i ].pipe_out[ EXECCMD_PIPE_WRITE ] );
closeWinHandle( &cmdtab[ i ].pipe_err[ EXECCMD_PIPE_READ ] );
closeWinHandle( &cmdtab[ i ].pipe_err[ EXECCMD_PIPE_WRITE ] );
string_renew( cmdtab[ i ].buffer_out );
string_renew( cmdtab[ i ].buffer_err );
}
}






static void invoke_cmd( char const * const command, int const slot )
{
SECURITY_ATTRIBUTES sa = { sizeof( SECURITY_ATTRIBUTES ), 0, 0 };
SECURITY_DESCRIPTOR sd;
STARTUPINFOA si = { sizeof( STARTUPINFOA ), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0 };


InitializeSecurityDescriptor( &sd, SECURITY_DESCRIPTOR_REVISION );
SetSecurityDescriptorDacl( &sd, TRUE, NULL, FALSE );
sa.lpSecurityDescriptor = &sd;
sa.bInheritHandle = TRUE;


string_new( cmdtab[ slot ].buffer_out );
string_new( cmdtab[ slot ].buffer_err );


if ( !CreatePipe( &cmdtab[ slot ].pipe_out[ EXECCMD_PIPE_READ ],
&cmdtab[ slot ].pipe_out[ EXECCMD_PIPE_WRITE ], &sa, IO_BUFFER_SIZE ) )
{
reportWindowsError( "CreatePipe", slot );
return;
}
if ( globs.pipe_action && !CreatePipe( &cmdtab[ slot ].pipe_err[
EXECCMD_PIPE_READ ], &cmdtab[ slot ].pipe_err[ EXECCMD_PIPE_WRITE ],
&sa, IO_BUFFER_SIZE ) )
{
reportWindowsError( "CreatePipe", slot );
return;
}


SetHandleInformation( cmdtab[ slot ].pipe_out[ EXECCMD_PIPE_READ ],
HANDLE_FLAG_INHERIT, 0 );
if ( globs.pipe_action )
SetHandleInformation( cmdtab[ slot ].pipe_err[ EXECCMD_PIPE_READ ],
HANDLE_FLAG_INHERIT, 0 );


si.dwFlags |= STARTF_USESHOWWINDOW;
si.wShowWindow = SW_HIDE;


si.dwFlags |= STARTF_USESTDHANDLES;
si.hStdOutput = cmdtab[ slot ].pipe_out[ EXECCMD_PIPE_WRITE ];
si.hStdError = globs.pipe_action
? cmdtab[ slot ].pipe_err[ EXECCMD_PIPE_WRITE ]
: cmdtab[ slot ].pipe_out[ EXECCMD_PIPE_WRITE ];


si.hStdInput = GetStdHandle( STD_INPUT_HANDLE );

if ( DEBUG_EXECCMD )
out_printf( "Command string for CreateProcessA(): '%s'\n", command );


if ( !CreateProcessA(
NULL                    ,  
(char *)command         ,  
NULL                    ,  
NULL                    ,  
TRUE                    ,  
CREATE_NEW_PROCESS_GROUP,  
NULL                    ,  
NULL                    ,  
&si                     ,  
&cmdtab[ slot ].pi ) )     
{
reportWindowsError( "CreateProcessA", slot );
return;
}

register_wait( slot );
}




static int raw_maxline()
{
if ( IsWindowsVersionOrGreater(5,0,0) == TRUE ) return 8191;  
if ( IsWindowsVersionOrGreater(4,0,0) == TRUE ) return 2047;  
return 996;                                      
}

static int maxline()
{
static int result;
if ( !result ) result = raw_maxline();
return result;
}




static void closeWinHandle( HANDLE * const handle )
{
if ( *handle )
{
CloseHandle( *handle );
*handle = 0;
}
}




static void string_renew( string * const s )
{
string_free( s );
string_new( s );
}




static long raw_command_length( char const * command )
{
char const * p;
char const * escape = 0;
char inquote = 0;
char const * newline = 0;


while ( isspace( *command ) )
++command;

p = command;


do
{
p += strcspn( p, "\n\"'<>|\\" );
switch ( *p )
{
case '\n':

newline = p;
while ( isspace( *++p ) );
if ( *p ) return -1;
break;

case '\\':
escape = escape && escape == p - 1 ? 0 : p;
++p;
break;

case '"':
case '\'':
if ( escape && escape == p - 1 )
escape = 0;
else if ( inquote == *p )
inquote = 0;
else if ( !inquote )
inquote = *p;
++p;
break;

case '<':
case '>':
case '|':
if ( !inquote )
return -1;
++p;
break;
}
}
while ( *p );


return ( newline ? newline : p ) - command;
}





#define add_carry_bit( a, b ) ((((a) | (b)) >> 31) & (~((a) + (b)) >> 31) & 0x1)


#define add_64_hi( h1, l1, h2, l2 ) ((h1) + (h2) + add_carry_bit(l1, l2))




static FILETIME add_64
(
unsigned long h1, unsigned long l1,
unsigned long h2, unsigned long l2
)
{
FILETIME result;
result.dwLowDateTime = l1 + l2;
result.dwHighDateTime = add_64_hi( h1, l1, h2, l2 );
return result;
}


static FILETIME add_FILETIME( FILETIME t1, FILETIME t2 )
{
return add_64( t1.dwHighDateTime, t1.dwLowDateTime, t2.dwHighDateTime,
t2.dwLowDateTime );
}


static FILETIME negate_FILETIME( FILETIME t )
{

return add_64( ~t.dwHighDateTime, ~t.dwLowDateTime, 0, 1 );
}




static double filetime_to_seconds( FILETIME const ft )
{
return ft.dwHighDateTime * ( (double)( 1UL << 31 ) * 2.0 * 1.0e-7 ) +
ft.dwLowDateTime * 1.0e-7;
}


static void record_times( HANDLE const process, timing_info * const time )
{
FILETIME creation;
FILETIME exit;
FILETIME kernel;
FILETIME user;
if ( GetProcessTimes( process, &creation, &exit, &kernel, &user ) )
{
time->system = filetime_to_seconds( kernel );
time->user = filetime_to_seconds( user );
timestamp_from_filetime( &time->start, &creation );
timestamp_from_filetime( &time->end, &exit );
}
}


static char ioBuffer[ IO_BUFFER_SIZE + 1 ];

#define FORWARD_PIPE_NONE 0
#define FORWARD_PIPE_STDOUT 1
#define FORWARD_PIPE_STDERR 2

static void read_pipe
(
HANDLE   in,  
string * out,
int      forwarding_mode
)
{
DWORD bytesInBuffer = 0;
DWORD bytesAvailable = 0;
DWORD i;

for (;;)
{

if ( !PeekNamedPipe( in, NULL, IO_BUFFER_SIZE, NULL,
&bytesAvailable, NULL ) || bytesAvailable == 0 )
return;


if ( !ReadFile( in, ioBuffer, bytesAvailable <= IO_BUFFER_SIZE ?
bytesAvailable : IO_BUFFER_SIZE, &bytesInBuffer, NULL ) || bytesInBuffer == 0 )
return;


for ( i = 0; i < bytesInBuffer; ++i )
{
if ( ( (unsigned char)ioBuffer[ i ] < 1 ) )
ioBuffer[ i ] = '?';
}

ioBuffer[ bytesInBuffer ] = '\0';

string_append( out, ioBuffer );

if ( forwarding_mode == FORWARD_PIPE_STDOUT )
out_data( ioBuffer );
else if ( forwarding_mode == FORWARD_PIPE_STDERR )
err_data( ioBuffer );
}
}

#define EARLY_OUTPUT( cmd ) \
( ! ( cmd.flags & EXEC_CMD_QUIET ) )

#define FORWARD_STDOUT( c )                                 \
( ( EARLY_OUTPUT( c ) && ( globs.pipe_action != 2 ) ) ? \
FORWARD_PIPE_STDOUT : FORWARD_PIPE_NONE )
#define FORWARD_STDERR( c )                                 \
( ( EARLY_OUTPUT( c ) && ( globs.pipe_action & 2 ) ) ?  \
FORWARD_PIPE_STDERR : FORWARD_PIPE_NONE )

static void read_output()
{
int i;
for ( i = 0; i < globs.jobs; ++i )
if ( cmdtab[ i ].pi.hProcess )
{

if ( cmdtab[ i ].pipe_out[ EXECCMD_PIPE_READ ] )
read_pipe( cmdtab[ i ].pipe_out[ EXECCMD_PIPE_READ ],
cmdtab[ i ].buffer_out, FORWARD_STDOUT( cmdtab[ i ] ) );

if ( cmdtab[ i ].pipe_err[ EXECCMD_PIPE_READ ] )
read_pipe( cmdtab[ i ].pipe_err[ EXECCMD_PIPE_READ ],
cmdtab[ i ].buffer_err, FORWARD_STDERR( cmdtab[ i ] ) );
}
}

static void CALLBACK try_wait_callback( void * data, BOOLEAN is_timeout )
{
struct _cmdtab_t * slot = ( struct _cmdtab_t * )data;
WaitForSingleObject( process_queue.write_okay, INFINITE );
process_queue.job_index = slot - cmdtab;
assert( !is_timeout );
SetEvent( process_queue.read_okay );

UnregisterWait( slot->wait_handle );
}

static int try_wait_impl( DWORD timeout )
{
int job_index;
int res = WaitForSingleObject( process_queue.read_okay, timeout );
if ( res != WAIT_OBJECT_0 )
return -1;
job_index = process_queue.job_index;
SetEvent( process_queue.write_okay );
return job_index;
}

static void register_wait( int job_id )
{
if ( globs.jobs > MAXIMUM_WAIT_OBJECTS )
{
RegisterWaitForSingleObject( &cmdtab[ job_id ].wait_handle,
cmdtab[ job_id ].pi.hProcess,
&try_wait_callback, &cmdtab[ job_id ], INFINITE,
WT_EXECUTEDEFAULT | WT_EXECUTEONLYONCE );
}
}



static int try_wait( int const timeoutMillis )
{
if ( globs.jobs <= MAXIMUM_WAIT_OBJECTS )
{
int i;
HANDLE active_handles[ MAXIMUM_WAIT_OBJECTS ];
int job_ids[ MAXIMUM_WAIT_OBJECTS ];
DWORD num_handles = 0;
DWORD wait_api_result;
for ( i = 0; i < globs.jobs; ++i )
{
if( cmdtab[ i ].pi.hProcess )
{
job_ids[ num_handles ] = i;
active_handles[ num_handles ] = cmdtab[ i ].pi.hProcess;
++num_handles;
}
}
wait_api_result = WaitForMultipleObjects( num_handles, active_handles, FALSE, timeoutMillis );
if ( WAIT_OBJECT_0 <= wait_api_result && wait_api_result < WAIT_OBJECT_0 + globs.jobs )
{
return job_ids[ wait_api_result - WAIT_OBJECT_0 ];
}
else
{
return -1;
}
}
else
{
return try_wait_impl( timeoutMillis );
}

}


static int try_kill_one()
{

if ( globs.timeout > 0 )
{
int i;
for ( i = 0; i < globs.jobs; ++i )
if ( cmdtab[ i ].pi.hProcess )
{
double const t = running_time( cmdtab[ i ].pi.hProcess );
if ( t > (double)globs.timeout )
{

close_alert( &cmdtab[ i ].pi );

kill_process_tree( cmdtab[ i ].pi.dwProcessId,
cmdtab[ i ].pi.hProcess );

return i;
}
}
}
return -1;
}


static void close_alerts()
{

if ( ( (float)clock() / (float)( CLOCKS_PER_SEC * 5 ) ) < ( 1.0 / 5.0 ) )
{
int i;
for ( i = 0; i < globs.jobs; ++i )
if ( cmdtab[ i ].pi.hProcess )
close_alert( &cmdtab[ i ].pi );
}
}




static double running_time( HANDLE const process )
{
FILETIME creation;
FILETIME exit;
FILETIME kernel;
FILETIME user;
if ( GetProcessTimes( process, &creation, &exit, &kernel, &user ) )
{

FILETIME current;
GetSystemTimeAsFileTime( &current );
return filetime_to_seconds( add_FILETIME( current,
negate_FILETIME( creation ) ) );
}
return 0.0;
}




static void kill_process_tree( DWORD const pid, HANDLE const process )
{
HANDLE const process_snapshot_h = CreateToolhelp32Snapshot(
TH32CS_SNAPPROCESS, 0 );
if ( INVALID_HANDLE_VALUE != process_snapshot_h )
{
BOOL ok = TRUE;
PROCESSENTRY32 pinfo;
pinfo.dwSize = sizeof( PROCESSENTRY32 );
for (
ok = Process32First( process_snapshot_h, &pinfo );
ok == TRUE;
ok = Process32Next( process_snapshot_h, &pinfo ) )
{
if ( pinfo.th32ParentProcessID == pid )
{

HANDLE const ph = OpenProcess( PROCESS_ALL_ACCESS, FALSE,
pinfo.th32ProcessID );
if ( ph )
{
kill_process_tree( pinfo.th32ProcessID, ph );
CloseHandle( ph );
}
}
}
CloseHandle( process_snapshot_h );
}

TerminateProcess( process, -2 );
}


static double creation_time( HANDLE const process )
{
FILETIME creation;
FILETIME exit;
FILETIME kernel;
FILETIME user;
return GetProcessTimes( process, &creation, &exit, &kernel, &user )
? filetime_to_seconds( creation )
: 0.0;
}




static int is_parent_child( DWORD const parent, DWORD const child )
{
HANDLE process_snapshot_h = INVALID_HANDLE_VALUE;

if ( !child )
return 0;
if ( parent == child )
return 1;

process_snapshot_h = CreateToolhelp32Snapshot( TH32CS_SNAPPROCESS, 0 );
if ( INVALID_HANDLE_VALUE != process_snapshot_h )
{
BOOL ok = TRUE;
PROCESSENTRY32 pinfo;
pinfo.dwSize = sizeof( PROCESSENTRY32 );
for (
ok = Process32First( process_snapshot_h, &pinfo );
ok == TRUE;
ok = Process32Next( process_snapshot_h, &pinfo ) )
{
if ( pinfo.th32ProcessID == child )
{

double tchild = 0.0;
double tparent = 0.0;
HANDLE const hchild = OpenProcess( PROCESS_QUERY_INFORMATION,
FALSE, pinfo.th32ProcessID );
CloseHandle( process_snapshot_h );



#ifdef UNICODE  
if ( !wcsicmp( pinfo.szExeFile, L"csrss.exe" ) &&
#else
if ( !stricmp( pinfo.szExeFile, "csrss.exe" ) &&
#endif
is_parent_child( parent, pinfo.th32ParentProcessID ) == 2 )
return 1;

#ifdef UNICODE  
if ( !wcsicmp( pinfo.szExeFile, L"smss.exe" ) &&
#else
if ( !stricmp( pinfo.szExeFile, "smss.exe" ) &&
#endif
( pinfo.th32ParentProcessID == 4 ) )
return 2;

if ( hchild )
{
HANDLE hparent = OpenProcess( PROCESS_QUERY_INFORMATION,
FALSE, pinfo.th32ParentProcessID );
if ( hparent )
{
tchild = creation_time( hchild );
tparent = creation_time( hparent );
CloseHandle( hparent );
}
CloseHandle( hchild );
}


if ( ( tchild == 0.0 ) || ( tparent == 0.0 ) ||
( tchild < tparent ) )
return 0;

return is_parent_child( parent, pinfo.th32ParentProcessID ) & 1;
}
}

CloseHandle( process_snapshot_h );
}

return 0;
}




BOOL CALLBACK close_alert_window_enum( HWND hwnd, LPARAM lParam )
{
char buf[ 7 ] = { 0 };
PROCESS_INFORMATION const * const pi = (PROCESS_INFORMATION *)lParam;
DWORD pid;
DWORD tid;


if (

!IsWindowVisible( hwnd )

|| !GetClassNameA( hwnd, buf, sizeof( buf ) )

|| strcmp( buf, "#32770" ) )
return TRUE;


tid = GetWindowThreadProcessId( hwnd, &pid );
if ( !tid || !is_parent_child( pi->dwProcessId, pid ) )
return TRUE;


PostMessageA( hwnd, WM_CLOSE, 0, 0 );


if ( WaitForSingleObject( pi->hProcess, 200 ) == WAIT_TIMEOUT )
{
PostThreadMessageA( tid, WM_QUIT, 0, 0 );
WaitForSingleObject( pi->hProcess, 300 );
}


return FALSE;
}


static void close_alert( PROCESS_INFORMATION const * const pi )
{
EnumWindows( &close_alert_window_enum, (LPARAM)pi );
}




static FILE * open_command_file( int const slot )
{
string * const command_file = cmdtab[ slot ].command_file;


if ( !command_file->value )
{
DWORD const procID = GetCurrentProcessId();
string const * const tmpdir = path_tmpdir();
string_new( command_file );
string_reserve( command_file, tmpdir->size + 64 );
command_file->size = sprintf( command_file->value,
"%s\\jam%lu-%02d-##.bat", tmpdir->value, procID, slot );
}


{
char * const index1 = command_file->value + command_file->size - 6;
char * const index2 = index1 + 1;
int waits_remaining;
assert( command_file->value < index1 );
assert( index2 + 1 < command_file->value + command_file->size );
assert( index2[ 1 ] == '.' );
for ( waits_remaining = 3; ; --waits_remaining )
{
int index;
for ( index = 0; index != 20; ++index )
{
FILE * f;
*index1 = '0' + index / 10;
*index2 = '0' + index % 10;
f = fopen( command_file->value, "w" );
if ( f ) return f;
}
if ( !waits_remaining ) break;
Sleep( 250 );
}
}

return 0;
}




static char const * prepare_command_file( string const * command, int slot )
{
FILE * const f = open_command_file( slot );
if ( !f )
{
err_printf( "failed to write command file!\n" );
exit( EXITBAD );
}
fputs( command->value, f );
fclose( f );
return cmdtab[ slot ].command_file->value;
}




static int get_free_cmdtab_slot()
{
int slot;
for ( slot = 0; slot < globs.jobs; ++slot )
if ( !cmdtab[ slot ].pi.hProcess )
return slot;
err_printf( "no slots for child!\n" );
exit( EXITBAD );
}




static void string_new_from_argv( string * result, char const * const * argv )
{
assert( argv );
assert( argv[ 0 ] );
string_copy( result, *(argv++) );
while ( *argv )
{
string_push_back( result, ' ' );
string_push_back( result, '"' );
string_append( result, *(argv++) );
string_push_back( result, '"' );
}
}




static void reportWindowsError( char const * const apiName, int slot )
{
char * errorMessage;
char buf[24];
string * err_buf;
timing_info time;
DWORD const errorCode = GetLastError();
DWORD apiResult = FormatMessageA(
FORMAT_MESSAGE_ALLOCATE_BUFFER |  
FORMAT_MESSAGE_FROM_SYSTEM |
FORMAT_MESSAGE_IGNORE_INSERTS,
NULL,                             
errorCode,                        
0,                                
(LPSTR)&errorMessage,             
0,                                
0 );                              


if ( globs.pipe_action )
err_buf = cmdtab[ slot ].buffer_err;
else
err_buf = cmdtab[ slot ].buffer_out;
string_append( err_buf, apiName );
string_append( err_buf, "() Windows API failed: " );
sprintf( buf, "%lu", errorCode );
string_append( err_buf, buf );

if ( !apiResult )
string_append( err_buf, ".\n" );
else
{
string_append( err_buf, " - " );
string_append( err_buf, errorMessage );

if( err_buf->value[ err_buf->size - 1 ] != '\n' )
string_push_back( err_buf, '\n' );
LocalFree( errorMessage );
}


time.system = 0;
time.user = 0;
timestamp_current( &time.start );
timestamp_current( &time.end );


(*cmdtab[ slot ].func)( cmdtab[ slot ].closure, EXEC_CMD_FAIL, &time,
cmdtab[ slot ].buffer_out->value, cmdtab[ slot ].buffer_err->value,
EXIT_OK );


closeWinHandle( &cmdtab[ slot ].pi.hProcess );
closeWinHandle( &cmdtab[ slot ].pi.hThread );
closeWinHandle( &cmdtab[ slot ].pipe_out[ EXECCMD_PIPE_READ ] );
closeWinHandle( &cmdtab[ slot ].pipe_out[ EXECCMD_PIPE_WRITE ] );
closeWinHandle( &cmdtab[ slot ].pipe_err[ EXECCMD_PIPE_READ ] );
closeWinHandle( &cmdtab[ slot ].pipe_err[ EXECCMD_PIPE_WRITE ] );
string_renew( cmdtab[ slot ].buffer_out );
string_renew( cmdtab[ slot ].buffer_err );
}


#endif 
