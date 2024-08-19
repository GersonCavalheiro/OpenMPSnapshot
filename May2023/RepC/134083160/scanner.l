D     [0-9]
L     [a-zA-Z_]
H     [a-fA-F0-9]
E     [Ee][+-]?{D}+
FS    (f|F|l|L)
IS    (u|U|l|L)*
%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "ompi.h"
#include "ast.h"            
#include "symtab.h"
#include "parser.h"
#include "scanner.h"
#define SCANNER_STR_SIZES 1024  
static int  on_omp_line = 0,    
on_ompix_line = 0;  
int         __has_omp = 0,      
__has_ompix = 0,
__has_affinitysched = 0;
void count(), sharp(), gobbleline(), gobblecomment();
int  checkomp(int type, int return_if_true), pragma_other();
#define IdentOrType() \
( symtab_get(stab,Symbol(yytext),TYPENAME) != NULL ? TYPE_NAME : IDENTIFIER )
static char origfile_name[SCANNER_STR_SIZES];  
static char thisfile_name[SCANNER_STR_SIZES];  
static int  thisfile_line = 1, 
marker_line = 0,   
origfile_line = 0; 
static int  thisfile_column = 0;  
static int  start_token = 0;   
%}
%%
%{
if (start_token)
{
int t = start_token;
start_token = 0;         
return t;
}
%}
"
"
[ \t]*"#"[ \t]*"pragma"[ \t]+"ompix"[ \t]+  {
count();
on_ompix_line = __has_ompix = 1;
return (PRAGMA_OMPIX);
}
[ \t]*"#"[ \t]*"pragma"[ \t]+"omp"[ \t]+  {
count();
on_omp_line = __has_omp = 1;
return (PRAGMA_OMP);
}
[ \t]*"#"[ \t]*"pragma"[ \t]+"omp"[ \t]+"threadprivate"[ \t]*  {
count();
on_omp_line = __has_omp = 1;
return (PRAGMA_OMP_THREADPRIVATE);
}
[ \t]*"#"[ \t]*"pragma"[ \t]+"omp"[ \t]+"cancellation"[ \t]+"point"[ \t]*  {
count();
on_omp_line = __has_omp = 1;
return (PRAGMA_OMP_CANCELLATIONPOINT);
}
[ \t]*"#"[ \t]*"pragma" {
return ( pragma_other() );
}
[ \t]*"#"[ \t]*"line"  {
sharp();
}
[ \t]*"#"              {
sharp();
}
"parallel"             { return checkomp(0, OMP_PARALLEL); }
"sections"             { return checkomp(0, OMP_SECTIONS); }
"nowait"               { return checkomp(0, OMP_NOWAIT); }
"ordered"              { return checkomp(0, OMP_ORDERED); }
"schedule"             { return checkomp(0, OMP_SCHEDULE); }
"dynamic"              { return checkomp(0, OMP_DYNAMIC); }
"guided"               { return checkomp(0, OMP_GUIDED); }
"runtime"              { return checkomp(0, OMP_RUNTIME); }
"auto"                 {
count();
if (on_omp_line)
return(OMP_AUTO);
else
return(AUTO);
}
"affinity"             { 
count();
if (on_omp_line)
{
__has_affinitysched = 1;
return(OMP_AFFINITY);
}
else
return IdentOrType();
}
"section"              { return checkomp(0, OMP_SECTION); }
"single"               { return checkomp(0, OMP_SINGLE); }
"master"               { return checkomp(0, OMP_MASTER); }
"critical"             { return checkomp(0, OMP_CRITICAL); }
"barrier"              { return checkomp(0, OMP_BARRIER); }
"atomic"               { return checkomp(0, OMP_ATOMIC); }
"flush"                { return checkomp(0, OMP_FLUSH); }
"private"              { return checkomp(0, OMP_PRIVATE); }
"firstprivate"         { return checkomp(0, OMP_FIRSTPRIVATE); }
"lastprivate"          { return checkomp(0, OMP_LASTPRIVATE); }
"shared"               { return checkomp(0, OMP_SHARED); }
"none"                 { return checkomp(0, OMP_NONE); }
"reduction"            { return checkomp(2, OMP_REDUCTION); }
"copyin"               { return checkomp(0, OMP_COPYIN); }
"num_threads"          { return checkomp(0, OMP_NUMTHREADS); }
"copyprivate"          { return checkomp(0, OMP_COPYPRIVATE); }
"task"                 { return checkomp(2, OMP_TASK); }
"untied"               { return checkomp(2, OMP_UNTIED); }
"taskwait"             { return checkomp(0, OMP_TASKWAIT); }
"collapse"             { return checkomp(0, OMP_COLLAPSE); }
"final"                { return checkomp(0, OMP_FINAL); }
"mergeable"            { return checkomp(0, OMP_MERGEABLE); }
"taskyield"            { return checkomp(0, OMP_TASKYIELD); }
"read"                 { return checkomp(0, OMP_READ); }
"write"                { return checkomp(0, OMP_WRITE); }
"capture"              { return checkomp(0, OMP_CAPTURE); }
"update"               { return checkomp(0, OMP_UPDATE); }
"min"                  { return checkomp(0, OMP_MIN); }
"max"                  { return checkomp(0, OMP_MAX); }
"proc_bind"            { return checkomp(0, OMP_PROCBIND); }
"close"                { return checkomp(0, OMP_CLOSE); }
"spread"               { return checkomp(0, OMP_SPREAD); }
"simd"                 { return checkomp(0, OMP_SIMD); }
"safelen"              { return checkomp(0, OMP_SAFELEN); }
"simdlen"              { return checkomp(0, OMP_SIMDLEN); }
"declare"              { return checkomp(0, OMP_DECLARE); }
"inbranch"             { return checkomp(0, OMP_INBRANCH); }
"notinbranch"          { return checkomp(0, OMP_NOTINBRANCH); }
"uniform"              { return checkomp(0, OMP_UNIFORM); }
"linear"               { return checkomp(0, OMP_LINEAR); }
"aligned"              { return checkomp(0, OMP_ALIGNED); }
"target"               { return checkomp(0, OMP_TARGET); }
"data"                 { return checkomp(0, OMP_DATA); }
"device"               { return checkomp(0, OMP_DEVICE); }
"map"                  { return checkomp(0, OMP_MAP); }
"alloc"                { return checkomp(0, OMP_ALLOC); }
"to"                   { return checkomp(0, OMP_TO); }
"from"                 { return checkomp(0, OMP_FROM); }
"tofrom"               { return checkomp(0, OMP_TOFROM); }
"end"                  { return checkomp(0, OMP_END); }
"teams"                { return checkomp(0, OMP_TEAMS); }
"distribute"           { return checkomp(0, OMP_DISTRIBUTE); }
"num_teams"            { return checkomp(0, OMP_NUMTEAMS); }
"thread_limit"         { return checkomp(0, OMP_THREADLIMIT); }
"dist_schedule"        { return checkomp(0, OMP_DISTSCHEDULE); }
"depend"               { return checkomp(0, OMP_DEPEND); }
"in"                   { return checkomp(2, OMP_IN); }
"out"                  { return checkomp(2, OMP_OUT); }
"inout"                { return checkomp(2, OMP_INOUT); }
"taskgroup"            { return checkomp(0, OMP_TASKGROUP); }
"seq_cst"              { return checkomp(0, OMP_SEQ_CST); }
"cancel"               { return checkomp(0, OMP_CANCEL); }
"initializer"          { return checkomp(0, OMP_INITIALIZER); }
"hint"                 { return checkomp(2, OMP_HINT); }
"source"               { return checkomp(0, OMP_SOURCE); }
"sink"                 { return checkomp(0, OMP_SINK); }
"release"              { return checkomp(0, OMP_RELEASE); }
"delete"               { return checkomp(0, OMP_DELETE); }
"always"               { return checkomp(0, OMP_ALWAYS); }
"enter"                { return checkomp(0, OMP_ENTER); }
"exit"                 { return checkomp(0, OMP_EXIT); }
"is_device_ptr"        { return checkomp(0, OMP_IS_DEVICE_PTR); }
"use_device_ptr"       { return checkomp(0, OMP_USE_DEVICE_PTR); }
"priority"             { return checkomp(0, OMP_PRIORITY); }
"taskloop"             { return checkomp(0, OMP_TASKLOOP); }
"threads"              { return checkomp(0, OMP_THREADS); }
"link"                 { return checkomp(0, OMP_LINK); }
"defaultmap"           { return checkomp(0, OMP_DEFAULTMAP); }
"scalar"               { return checkomp(0, OMP_SCALAR); }
"monotonic"            { return checkomp(0, OMP_MONOTONIC); }
"nonmonotonic"         { return checkomp(0, OMP_NONMONOTONIC); }
"primary"               { return checkomp(0, OMP_PRIMARY); }
"taskdef"              { return checkomp(1, OMPIX_TASKDEF); }
"uponreturn"           { return checkomp(1, OMPIX_UPONRETURN); }
"tasksync"             { return checkomp(1, OMPIX_TASKSYNC); }
"IN"|"In"              { return checkomp(1, OMP_IN); }
"OUT"|"Out"            { return checkomp(1, OMP_OUT); }
"INOUT"|"Inout"|"InOut" {
return checkomp(1, OMP_INOUT);
}
"atnode"|"ATNODE"|"Atnode"|"AtNode" {
return checkomp(1, OMPIX_ATNODE);
}
"atworker"|"ATWORKER"|"Atworker"|"AtWorker" {
return checkomp(1, OMPIX_ATWORKER);
}
"taskschedule"         { return checkomp(1, OMPIX_TASKSCHEDULE); }
"stride"               { return checkomp(1, OMPIX_STRIDE); }
"start"                { return checkomp(1, OMPIX_START); }
"scope"                { return checkomp(1, OMPIX_SCOPE); }
"nodes"                { return checkomp(1, OMPIX_NODES); }
"workers"              { return checkomp(1, OMPIX_WORKERS); }
"local"                { return checkomp(1, OMPIX_LOCAL); }
"here"                 { return checkomp(1, OMPIX_HERE); }
"remote"               { return checkomp(1, OMPIX_REMOTE); }
"global"               { return checkomp(1, OMPIX_GLOBAL); }
"tied"                 { return checkomp(1, OMPIX_TIED); }
"detached"|"DETACHED"|"Detached" {
return checkomp(1, OMPIX_DETACHED);
}
"break"                { count(); return(BREAK); }
"case"                 { count(); return(CASE); }
"char"                 { count(); return(CHAR); }
"const"                { count(); return(CONST); }
"continue"             { count(); return(CONTINUE); }
"default"              {
count();
if (on_omp_line)
return(OMP_DEFAULT);
else
return (DEFAULT);
}
"do"                   { count(); return(DO); }
"double"               { count(); return(DOUBLE); }
"else"                 { count(); return(ELSE); }
"enum"                 { count(); return(ENUM); }
"extern"               { count(); return(EXTERN); }
"float"                { count(); return(FLOAT); }
"for"                  {
count();
if (on_omp_line)
return (OMP_FOR);
else
return(FOR);
}
"goto"                 { count(); return(GOTO); }
"if"                   {
count();
if (on_omp_line)
return(OMP_IF);
else
return(IF);
}
"int"                  { count(); return(INT); }
"long"                 { count(); return(LONG); }
"register"             { count(); return(REGISTER); }
"return"               { count(); return(RETURN); }
"short"                { count(); return(SHORT); }
"signed"               { count(); return(SIGNED); }
"sizeof"               { count(); return(SIZEOF); }
"static"               {
count();
if (on_omp_line)
return(OMP_STATIC);
else
return (STATIC);
}
"struct"               { count(); return(STRUCT); }
"switch"               { count(); return(SWITCH); }
"typedef"              { count(); return(TYPEDEF); }
"union"                { count(); return(UNION); }
"unsigned"             { count(); return(UNSIGNED); }
"_Bool"                { count(); return(UBOOL); }
"_Complex"             { count(); return(UCOMPLEX); }
"_Imaginary"           { count(); return(UIMAGINARY); }
"void"                 { count(); return(VOID); }
"restrict"             { count(); return(RESTRICT); }
"volatile"             { count(); return(VOLATILE); }
"inline"               { count(); return(INLINE); }
"while"                { count(); return(WHILE); }
"__builtin_va_arg"     { count(); return(__BUILTIN_VA_ARG); }
"__builtin_offsetof"   { count(); return(__BUILTIN_OFFSETOF); }
"__builtin_types_compatible_p" { count();return(__BUILTIN_TYPES_COMPATIBLE_P); }
"__attribute__"        { count(); return(__ATTRIBUTE__); }
"__volatile__"         { count(); return(VOLATILE);  }
"__asm__"              { count(); return(__ASM__); }
"__asm"                { count(); return(__ASM__); }
"asm"                  { count(); return(__ASM__); }
{L}({L}|{D})*          { count(); return( IdentOrType() ); }
0[xX]{H}+{IS}?         { count(); return(CONSTANT); }
0{D}+{IS}?             { count(); return(CONSTANT); }
{D}+{IS}?              { count(); return(CONSTANT); }
'(\\.|[^\\'])+'        { count(); return(CONSTANT); }
{D}+{E}{FS}?           { count(); return(CONSTANT); }
{D}*"."{D}+({E})?{FS}? { count(); return(CONSTANT); }
{D}+"."{D}*({E})?{FS}? { count(); return(CONSTANT); }
\"(\\.|[^\\"])*\"      { count(); return(STRING_LITERAL); }
">>="                  { count(); return(RIGHT_ASSIGN); }
"<<="                  { count(); return(LEFT_ASSIGN); }
"+="                   { count(); return(ADD_ASSIGN); }
"-="                   { count(); return(SUB_ASSIGN); }
"*="                   { count(); return(MUL_ASSIGN); }
"/="                   { count(); return(DIV_ASSIGN); }
"%="                   { count(); return(MOD_ASSIGN); }
"&="                   { count(); return(AND_ASSIGN); }
"^="                   { count(); return(XOR_ASSIGN); }
"|="                   { count(); return(OR_ASSIGN); }
">>"                   { count(); return(RIGHT_OP); }
"<<"                   { count(); return(LEFT_OP); }
"++"                   { count(); return(INC_OP); }
"--"                   { count(); return(DEC_OP); }
"->"                   { count(); return(PTR_OP); }
"&&"                   { count(); return(AND_OP); }
"||"                   { count(); return(OR_OP); }
"<="                   { count(); return(LE_OP); }
">="                   { count(); return(GE_OP); }
"=="                   { count(); return(EQ_OP); }
"!="                   { count(); return(NE_OP); }
";"                    { count(); return(';'); }
"{"                    { count(); return('{'); }
"}"                    { count(); return('}'); }
","                    { count(); return(','); }
":"                    { count(); return(':'); }
"="                    { count(); return('='); }
"("                    { count(); return('('); }
")"                    { count(); return(')'); }
"["                    { count(); return('['); }
"]"                    { count(); return(']'); }
"."                    { count(); return('.'); }
"&"                    { count(); return('&'); }
"!"                    { count(); return('!'); }
"~"                    { count(); return('~'); }
"-"                    { count(); return('-'); }
"+"                    { count(); return('+'); }
"*"                    { count(); return('*'); }
"/"                    { count(); return('/'); }
"%"                    { count(); return('%'); }
"<"                    { count(); return('<'); }
">"                    { count(); return('>'); }
"^"                    { count(); return('^'); }
"|"                    { count(); return('|'); }
"?"                    { count(); return('?'); }
"..."                  { count(); return(ELLIPSIS); }
[ \t\v\f]              { count(); }
\n                     {
count();
if (on_omp_line || on_ompix_line)
{
on_omp_line = on_ompix_line = 0;  
return('\n');
}
}
.                      {  }
%%
int yywrap()
{
return(1);
}
void sharp()
{
char c, line[SCANNER_STR_SIZES] = {0}, *s, *t;
int  lineno = -1;
for (s = line; (c = input()) != '\n' && c != 0; )   
if (s-line < SCANNER_STR_SIZES-2)
*s++ = c;
if (c == 0) return;          
*s = 0;                      
thisfile_line++;             
thisfile_column = 0;
if ( sscanf(line, "%d", &lineno) < 0 ) return;   
for (s = line; *s != '\"' && *s != 0; s++)       
;
if (*s == 0) return;                             
for (t = (++s); *s != '\"' && *s != 0; s++)      
;
if (t == s) return;                              
*s = 0;
strncpy(origfile_name, t, SCANNER_STR_SIZES-1);
origfile_line = lineno;
marker_line   = thisfile_line;
}
int pragma_other()
{
char c, line[SCANNER_STR_SIZES] = {0}, *s, *t;
int  lineno = -1;
sprintf(line, "#pragma");
for (s = line+7; (c = input()) != '\n' && c != 0; )   
if (s-line < SCANNER_STR_SIZES-2)
*s++ = c;
*s = 0;                      
strcpy(yylval.name, line);
if (c != 0)                  
{
thisfile_line++;           
thisfile_column = 0;
}
return (PRAGMA_OTHER);
}
void gobbleline()
{
char c;
for ( ; (c = input()) != '\n' && c != 0; )   
;
thisfile_line++;             
thisfile_column = 0;
}
char gobblespaces()
{
char c;
for ( ; (c = input()) != 0 && isspace(c); )
if (c == '\n')
{
thisfile_line++;
thisfile_column = 0;
}
else
thisfile_column++;
if (c)
thisfile_column++;
return (c);
}
void gobblecomment()
{
char c, done = 0;
for ( ; !done && (c = input()) != 0; )
{
if (c == '\n')
{
thisfile_line++;               
thisfile_column = 0;
continue;
}
thisfile_column++;
if (c == '*')
{
while ((c = input()) == '*')
thisfile_column++;
if (c == '\n')
{
thisfile_line++;             
thisfile_column = 0;
}
else
{
thisfile_column++;
if (c == '/')
done = 1;
}
}
}
}
int sc_scan_attribute(char **string)
{
char c, done = 0, text[SCANNER_STR_SIZES];
int  n;
c = gobblespaces();
if (c != '(' || ((c = input()) != '(')) return (-1);
thisfile_column++;
text[0] = text[1] = '(';
for (n = 2; !done && (c = input()) != 0 && n < SCANNER_STR_SIZES; )
{
text[n++] = c;
if (c == '\n')
{
thisfile_line++;               
thisfile_column = 0;
continue;
}
thisfile_column++;
if (c == ')')
{
if ((c = input()) == ')')
done = 1;
text[n++] = c;
if (done)
{
text[n++] = 0;
*string = strdup(text);
}
thisfile_column++;
}
}
return ( (c == 0 || n >= SCANNER_STR_SIZES) ? -1 : 0 );
}
void count()
{
int i, nonempty = 0;
for (i = 0; yytext[i] != 0; i++)
if (yytext[i] == '\n')
{
thisfile_column = 0;
thisfile_line++;
}
else
if (yytext[i] == '\t')
thisfile_column += ( 8 - (thisfile_column % 8) );
else
{
thisfile_column++;
if (!isspace(yytext[i]))
nonempty = 1;
};
if (nonempty)
strcpy(yylval.name, yytext);
}
int checkomp(int type, int return_if_true) {
count();
if ((type == 0 && on_omp_line) ||
(type == 1 && on_ompix_line) ||
(type == 2 && (on_omp_line || on_ompix_line)))
return(return_if_true);
else
return IdentOrType();
}
void sc_scan_string(char *s)
{
yy_scan_string(s);
*origfile_name = 0;
sc_set_filename("injected_code");
}
void sc_set_start_token(int t)
{ start_token = t; thisfile_line = 1; thisfile_column = 0;
marker_line = origfile_line = 0; }
void sc_set_filename(char *fn)
{ strncpy(thisfile_name, fn, 255); }
char *sc_original_file()
{ return ( (*origfile_name) ?  origfile_name : thisfile_name ); }
int  sc_original_line()  { return(thisfile_line - marker_line + origfile_line);}
int  sc_line()           { return(thisfile_line); }
int  sc_column()         { return(thisfile_column); }
void sc_pause_openmp()   { on_omp_line = 0; }
void sc_start_openmp()   { on_omp_line = 1; }
