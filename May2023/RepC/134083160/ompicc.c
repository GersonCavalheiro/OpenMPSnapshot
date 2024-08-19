#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#if defined(HAVE_REALPATH)
#include <limits.h>
#endif
#include "config.h"
#include "stddefs.h"
#include "git_version.h"
#include "str.h"
#include "keyval.h"
#include "ompicc.h"
#include "kernels.h"
int myeecb, ort;
int sysexec(char *cmd, int *exit_flag)
{
int ret;
if ((ret = system(cmd)) == -1)
{
if (exit_flag)
return (*exit_flag = -1);
_exit(-1);
}
exit_flag && (*exit_flag = 0);
return (WIFEXITED(ret) ? WEXITSTATUS(ret) : ret);
}
char PREPROCESSOR[PATHSIZE], COMPILER[PATHSIZE],
CPPFLAGS[FLAGSIZE], CFLAGS[FLAGSIZE], LDFLAGS[FLAGSIZE],
ORTINFO[FLAGSIZE];
char ortlibname[PATHSIZE],           
RealOmpiName[PATHSIZE];
#ifdef PORTABLE_BUILD
char InstallPath[PATHSIZE], LibDir[PATHSIZE], IncludeDir[PATHSIZE];
#endif
#define ompi_info() \
fprintf(stderr,\
"This is %s %s using\n  >> system compiler: %s\n  >> runtime library: %s\n"\
"  >> config. devices: %s\n",\
PACKAGE_NAME, GIT_VERSION, COMPILER, *ORTINFO ? ORTINFO : ortlibname,\
MODULES_CONFIG)
static void ompicc_error(int exitcode, char *format, ...)
{
va_list ap;
va_start(ap, format);
fprintf(stderr, "[ompicc error]: ");
vfprintf(stderr, format, ap);
va_end(ap);
exit(exitcode);
}
static char *get_basename(char *path)
{
char *s;
if (path == NULL || *path == 0)
return ".";
else
if (path[0] == '/' && path[1] == 0)
return path;
else
{
s = path;
while (*s)
s++;
s--;
while (*s == '/' && s != path)   
*s-- = 0;
if (s == path)
return path;
while (*s != '/' && s != path)
s--;
if (*s == '/')
return s + 1;
else return path;
}
}
static arg_t *new_arg(char opt, char *val)
{
arg_t *p;
if ((p = (arg_t *) malloc(sizeof(arg_t))) == NULL)
ompicc_error(-1, "malloc() failed\n");
p->opt = opt;
if (val != NULL)
strcpy(p->val, val);
else p->val[0] = 0;
p->next = NULL;
return p;
}
void arglist_add(arglist_t *l, arg_t *arg)
{
if (l->head == NULL)
l->head = l->tail = arg;
else
{
l->tail->next = arg;
l->tail = arg;
}
}
int append_arg(arglist_t *l, int argc, char **argv, int proceed)
{
char opt, val[SLEN];
arg_t *p;
val[0] = 0;
if (argv[0][0] == 0)
return 0;
if (argv[0][0] != '-')
{
p = new_arg(0, *argv);
arglist_add(l, p);
return 0;
}
opt = argv[0][1];
if (argv[0][2] != 0)
{
strcpy(val, &argv[0][2]);
p = new_arg(opt, val);
arglist_add(l, p);
return 0;
}
else
{
if (proceed && argc > 1)
strcpy(val, &argv[1][0]);
p = new_arg(opt, val);
arglist_add(l, p);
return proceed && argc > 1;
}
}
static int quotedlen(char *s)
{
int len;
for (len = 0; *s != 0; s++)
len += (*s == '"' ? 2 : 1);
return (len);
}
static void strarglist(char *dest, arglist_t *l, int maxlen)
{
arg_t *p;
char  *c, *d = dest;
for (*d = 0, p = l->head; p != NULL; p = p->next)
{
if ((d - dest) + quotedlen(p->val) + 6 >= maxlen)
ompicc_error(1, "argument(s) too long; rebuild OMPi with larger LEN.\n");
if (p->opt)
{
snprintf(dest, maxlen, "-%c ", p->opt);
dest += ((p->opt == 'o') ? 3 : 2);
}
*(dest++) = '"';
for (c = p->val; *c != 0; c++)
{
if (*c == '"') *(dest++) = '\\';
*(dest++) = *c;
}
*(dest++) = '"';
*(dest++) = ' ';
}
*dest = 0;
}
int fok(char *fname)
{
struct stat buf;
return (stat(fname, &buf) == 0);
}
bool disableOpenMP = false,
disableOmpix = false,
cppLineNo = false;
bool mustlink = true;     
bool makefile = true;     
int  keep = 0;            
bool verbose = false;
bool showdbginfo = false; 
bool showdevinfo = false; 
bool longdevinfo = false; 
bool usegdb = false;      
bool usecarstats = false;
bool reductionOld = false; 
bool autoscope = false;    
int  taskoptlevel = -1;    
char *reqmodules = NULL;   
arglist_t user_files     = { NULL, NULL };  
arglist_t user_outfile   = { NULL, NULL };  
arglist_t user_prep_args = { NULL, NULL };  
arglist_t user_link_args = { NULL, NULL };  
arglist_t user_scc_flags = { NULL, NULL };  
#define OPTNAME(opt)   "--" #opt
#define OPTNAME_V(opt) "V--" #opt "="
#define OPTION(opt)    OPT_##opt
typedef enum {
OPTION(unknown) = -1, 
OPTION(version) = 0, OPTION(options),     OPTION(envs),
OPTION(ort),         OPTION(nomp),        OPTION(nox),
OPTION(taskopt),     OPTION(nomakefile),  OPTION(nolineno),
OPTION(gdb),         OPTION(dbg),         OPTION(devs),
OPTION(devinfo),     OPTION(devvinfo),    OPTION(reduction),
OPTION(autoscope),   OPTION(carstats),
OPTION(lastoption)    
} option_t;
char *optnames[] = {
OPTNAME(version),    OPTNAME(options),     OPTNAME(envs),
OPTNAME_V(ort),      OPTNAME(nomp),        OPTNAME(nox),
OPTNAME_V(taskopt),  OPTNAME(nomakefile),  OPTNAME(nolineno),
OPTNAME(gdb),        OPTNAME(dbg),         OPTNAME_V(devs),
OPTNAME(devinfo),    OPTNAME(devvinfo),    OPTNAME_V(reduction),
OPTNAME(autoscope),  OPTNAME(carstats)
};
char *optinfo[] = {
"", "    (show all OMPi options)", "    (show all available env vars)",
"<eelib>",   "", "",
"[size|speed|speed+]    (default: speed+)", "", "    (produces no # <line>)",
"    (runs _ompi from within gdb)", "", "<deviceid,deviceid,...>",
"     (show short devices info)", "    (show long devices info)", "[old|new]",
"   (enable autoscoping analysis)", "    (turn on C.A.R.S. analysis)"
};
option_t optid(char *arg, char **val)
{
int i;
for (i = 0; i < OPTION(lastoption); i++)
if (optnames[i][0] == 'V')     
{
if (strncmp(optnames[i]+1, arg, strlen(optnames[i])-1) == 0)
{
*val = arg + strlen(optnames[i]) - 1;
return ((option_t) i);
}
}
else
if (strcmp(optnames[i], arg) == 0)
return ((option_t) i);
return ( OPTION(unknown) );
}
void showallopts()
{
int i;
printf("all available options\n---------------------\n");
for (i = 0; i < OPTION(lastoption); i++)
printf("  %s%s\n", optnames[i]+(optnames[i][0] == 'V' ? 1 : 0), optinfo[i]);
}
void helpOpenMPenv()
{
int i, max = 0;
char *ompenv[][2] = {
{ "  OMP_DYNAMIC",           "boolean" },
{ "  OMP_NESTED",            "boolean" },
{ "  OMP_SCHEDULE",          "policy[,int]" },
{ "",                        "     policy=static|dynamic|guided|auto" },
{ "  OMP_STACKSIZE",         "int[C]     C=B|K|M|G (default:K)" },
{ "  OMP_THREAD_LIMIT",      "int" },
{ "  OMP_MAX_ACTIVE_LEVELS", "int" },
{ "  OMP_WAIT_POLICY",       "active|passive" },
{ "  OMP_NUM_THREADS",       "int[,int[,int ...]]" },
{ "  OMP_PROC_BIND",         "true|false|<list of types>" },
{ "",                        "     types=master|close|spread" },
{ "  OMP_CANCELLATION",      "boolean" },
{ "  OMP_DISPLAY_ENV",       "true|false|verbose" },
{ "  OMP_PLACES",            "symbolic[(int)]|<list of places>" },
{ "",                        "     symbolic=thread|cores|sockets" },
{ "  OMP_DEFAULT_DEVICE",    "int" },
{ "  OMP_MAX_TASK_PRIORITY", "int" },
{ "  OMP_DISPLAY_AFFINITY",  "boolean" },
{ "  OMP_AFFINITY_FORMAT",   "string" },
{ "  OMP_TARGET_OFFLOAD",    "MANDATORY|DISABLED|DEFAULT" },
{ "  OMPI_DYNAMIC_TASKQUEUESIZE", "boolean     (defualt:false)" },
{ "  OMPI_STEAL_POLICY",     "FIFO|LIFO   (default:FIFO)" },
{ "  OMPI_PAR2TASK_POLICY",  "true|false|auto  (default:auto)" },
{ "  OMPI_HOSTTARGET_SHARE", "boolean     (default:false)" },
{ NULL, NULL }
};
printf("all runtime env. variables\n--------------------------\n");
for (i = 0; ompenv[i][0] != NULL; i++)
if (strlen(ompenv[i][0]) > max)
max = strlen(ompenv[i][0]);
for (i = 0; ompenv[i][0] != NULL; i++)
printf("%-*s  %s\n", max, ompenv[i][0], ompenv[i][1]);
}
void parse_args(int argc, char **argv)
{
int  d, ortlib = 0;
char *parameter, *val;
argv++;
argc--;
while (argc)
{
d = 0;
switch ( optid(parameter = argv[0], &val) )
{
case OPTION(version):
printf("%s\n", GIT_VERSION);
_exit(0);
break;
case OPTION(options):
showallopts();
_exit(0);
break;
case OPTION(envs):
helpOpenMPenv();
_exit(0);
break;
case OPTION(ort):
strncpy(ortlibname, val, 511);
ortlib = 1;
break;
case OPTION(nomp):       disableOpenMP = true; break;
case OPTION(nox):        disableOmpix = true; break;
case OPTION(nomakefile): makefile = false; break;
case OPTION(nolineno):   cppLineNo = true; break;
case OPTION(gdb):        usegdb = true; break;
case OPTION(dbg):        showdbginfo = true; break;
case OPTION(carstats):   usecarstats = true; break;
case OPTION(devinfo):    showdevinfo = true; longdevinfo = false; break;
case OPTION(devvinfo):   showdevinfo = true; longdevinfo = true; break;
case OPTION(autoscope):  autoscope = true; break;
case OPTION(taskopt):
if (strcmp(val, "size") == 0)
taskoptlevel = 0;
else
if (strcmp(val, "speed") == 0)
taskoptlevel = 1;
else
if (strcmp(val, "speed+") == 0)
taskoptlevel = 2;
break;
case OPTION(devs):
reqmodules = val;
break;
case OPTION(reduction):
if (strcmp(val, "old") == 0)
reductionOld = true;
else
if (strcmp(val, "new"))
ompicc_error(1,"unknown reduction request (try 'new' or 'old').\n");
break;
default:
if (parameter[0] == '-')              
switch (parameter[1])
{
case 'c':
mustlink = false;
break;
case 'l':
d = append_arg(&user_link_args, argc, argv, 1);
break;
case 'L':
d = append_arg(&user_link_args, argc, argv, 1);
break;
case 'I':
d = append_arg(&user_prep_args, argc, argv, 1);
break;
case 'D':
d = append_arg(&user_prep_args, argc, argv, 1);
break;
case 'U':
d = append_arg(&user_prep_args, argc, argv, 1);
break;
case 'o':
d = append_arg(&user_outfile, argc, argv, 1);
break;
case 'k':
keep = 1; d = 0;
break;
case 'K':
keep = 2; d = 0;
break;
case 'v':
verbose = true; d = 0;
break;
default:
d = append_arg(&user_scc_flags, argc, argv, 0);
break;
}
else
{
d = append_arg(&user_files, argc, argv, 0);
if (!fok(user_files.tail->val))
ompicc_error(1, "file %s does not exist\n", user_files.tail->val);
};
}
argc = argc - 1 - d;
argv = argv + 1 + d;
}
if (!ortlib)
strcpy(ortlibname, "default");    
}
void ompicc_compile(char *fname)
{
char *s, *compfmt, preflags[SLEN], noext[PATHSIZE], outfile[PATHSIZE];
char cmd[LEN], strscc_flags[LEN], strgoutfile[LEN];
int  res, eflag;
int  nkernels = 0;  
if ((s = strrchr(fname, '.')) != NULL)
if (strcmp(s, ".o") == 0) return;     
strcpy(noext, fname);
if ((s = strrchr(noext, '.')) != NULL) *s = 0; 
snprintf(outfile, PATHSIZE, "%s_ompi.c", noext);
strarglist(preflags, &user_prep_args, SLEN);
#if defined(__SYSOS_cygwin) && defined(__SYSCOMPILER_cygwin)
snprintf(cmd, LEN, "%s -U__CYGWIN__ -D__extension__=  -U__GNUC__ "
" %s -I%s %s \"%s\" > \"%s.pc\"",
PREPROCESSOR, CPPFLAGS, IncludeDir, preflags, fname, noext);
#else
snprintf(cmd, LEN, "%s -U__GNUC__ %s -I%s %s \"%s\" > \"%s.pc\"",
PREPROCESSOR, CPPFLAGS, IncludeDir, preflags, fname, noext);
#endif
if (verbose)
fprintf(stderr, "====> Preprocessing file (%s.c)\n  [ %s ]\n", noext, cmd);
if ((res = sysexec(cmd, NULL)) != 0)
_exit(res);
#ifdef PORTABLE_BUILD
compfmt = "%s%s%s \"%s.pc\" __ompi__ \"%s\"%s%s%s%s%s%s%s%s%s%s %s > \"%s\"%s";
#else
compfmt = "%s%s%s \"%s.pc\" __ompi__%s%s%s%s%s%s%s%s%s%s %s > \"%s\"%s";
#endif
snprintf(cmd, LEN, compfmt,
usegdb ? "gdb " : "", 
RealOmpiName,
usegdb ? " -ex 'set args" : "", 
noext,
#ifdef PORTABLE_BUILD
InstallPath,             
#endif
strstr(CFLAGS, "OMPI_MAIN=LIB") ? " -nomain " : "",
strstr(CFLAGS, "OMPI_MEMMODEL=PROC") == NULL ? "" :
strstr(CFLAGS, "OMPI_MEMMODEL=THR") ? " -procs -threads " :
" -procs ",
disableOpenMP ? " -nomp " : "",
disableOmpix ? " -nox " : "",
taskoptlevel == 0 ? " -taskopt0 " : 
taskoptlevel == 2 ? " -taskopt2 " : " -taskopt1 ",
cppLineNo ? " -nolineno " : "",
showdbginfo ? " -showdbginfo " : "",
usecarstats ? " -drivecar " : "",
reductionOld ? " -oldred " : "",
autoscope ? " -autoscope" : "",
modules_argfor_ompi(),
outfile,
usegdb ? "'" : "");
if (verbose)
fprintf(stderr, "====> Transforming file (%s.c)\n  [ %s ]\n", noext, cmd);
res = sysexec(cmd, NULL);
if (keep < 2)
{
snprintf(cmd, LEN, "%s.pc", noext);               
unlink(cmd);
}
if (res == 33)                                
{
FILE *of = fopen(outfile, "w");
if (of == NULL)
{
fprintf(stderr, "Cannot write to intermediate file.\n");
_exit(1);
}
if (cppLineNo)
fprintf(of, "# 1 \"%s.c\"\n", noext); 
fclose(of);
snprintf(cmd, LEN, "cat \"%s.c\" >> \"%s\"", noext, outfile);
if (sysexec(cmd, &eflag))
{
unlink(outfile);
_exit(res);
}
}
else
{
if (res != 0)
{
if (!keep)
unlink(outfile);
_exit(res);
}
else   
{
FILE *of = fopen(outfile, "r");
char ch = 0;
if (of != NULL)
{
for (ch = fgetc(of); ch!=EOF && ch!='\n'; ch = fgetc(of))
;
if (ch == '\n')
fscanf(of, "$OMPi__nfo:%d", &nkernels);
fclose(of);
}
}
}
strarglist(strscc_flags, &user_scc_flags, LEN);
snprintf(cmd, LEN, "%s \"%s\" -c %s -I%s %s %s",
COMPILER, outfile, CFLAGS, IncludeDir, preflags, strscc_flags);
if (verbose)
fprintf(stderr, "====> Compiling file (%s):\n  [ %s ]\n", outfile, cmd);
res = sysexec(cmd, &eflag);
if (!keep)
unlink(outfile);
if (eflag || res)
_exit(res);
strarglist(strgoutfile, &user_outfile, LEN);
strcpy(noext, get_basename(fname));
if ((s = strrchr(noext, '.')) != NULL) * s = 0; 
if (user_outfile.head != NULL && !mustlink)
strcpy(outfile, user_outfile.head->val);
else
snprintf(outfile, PATHSIZE, "%s.o", noext);
strcat(noext, "_ompi.o");
if (verbose)
fprintf(stderr, "====> Renaming file \"%s\" to \"%s\"\n",
noext, outfile);
rename(noext, outfile);
if (nkernels && nmodules)
{
if (verbose)
fprintf(stderr, "====> Generating kernel makefiles for %d module(s)\n",
nmodules);
kernel_makefiles(fname, nkernels); 
}
}
void ompicc_link()
{
arg_t *p;
char cur_obj[PATHSIZE], tmp[PATHSIZE], cmd[LEN];
char objects[LEN], *obj;
char strsccargs[LEN], strlinkargs[LEN], strgoutfile[LEN], strprepargs[LEN];
int len, is_tmp, eflag;
char rm_obj[LEN];
obj = objects;
*obj = 0;
strcpy(rm_obj, "rm -f ");
for (p = user_files.head; p != NULL; p = p->next)
{
strcpy(cur_obj, p->val);
is_tmp = 0;
len = strlen(cur_obj);
if (cur_obj[len - 1] == 'c')
is_tmp = 1;
if (is_tmp)
{
cur_obj[len - 2] = 0;
strcpy(tmp, cur_obj);
strcpy(cur_obj, get_basename(tmp));
strcat(cur_obj, ".o");
strcat(rm_obj, cur_obj);
strcat(rm_obj, " ");
}
snprintf(obj, LEN, "\"%s\" ", cur_obj);
obj += strlen(cur_obj) + 3;
}
strarglist(strsccargs,  &user_scc_flags, LEN);
strarglist(strlinkargs, &user_link_args, LEN);
strarglist(strgoutfile, &user_outfile, LEN);
strarglist(strprepargs, &user_prep_args, LEN);
snprintf(cmd, LEN,
"%s %s %s -I%s %s %s %s -L%s -L%s/%s -lort %s %s -lort",
COMPILER, objects, CFLAGS, IncludeDir, strprepargs, strsccargs,
strgoutfile, LibDir, LibDir, ortlibname, LDFLAGS, strlinkargs);
if (verbose)
fprintf(stderr, "====> Linking:\n  [ %s ]\n", cmd);
if (sysexec(cmd, &eflag))
fprintf(stderr, "Error: could not perform linking.\n");
sysexec(rm_obj, &eflag);   
}
void replace_variable(char *line, char *variable, char *replace_with)
{
char *p, tmp[SLEN];
while (p = strstr(line, variable))
{
snprintf(tmp, SLEN, "%s%s", replace_with, p + strlen(variable));
strcpy(p, tmp);
}
}
static void getargs(char *dest, arglist_t *l, int maxlen)
{
arg_t *p;
char  *c, *d = dest;
for (*d = 0, p = l->head; p != NULL; p = p->next)
{
if ((d - dest) + quotedlen(p->val) + 6 >= maxlen)
ompicc_error(1, "argument(s) too long; rebuild OMPi with larger LEN.\n");
for (c = p->val; *c != 0; c++)
{
if (*c == '"') *(dest++) = '\\';
*(dest++) = *c;
}
if (p->next)
*(dest++) = ' ';
}
*dest = 0;
}
void ompicc_makefile(char *compilerPath)
{
char strfiles[LEN], strgoutfile[LEN];
char *s, preflags[SLEN], noext[PATHSIZE], infile[PATHSIZE],
outfile[PATHSIZE], line[SLEN];
FILE *fp, *infp;
snprintf(infile, PATHSIZE, "%s/%s/MakefileTemplate", LibDir, ortlibname);
if (!(infp = fopen(infile, "r")))
{
makefile = false;
return;
}
getargs(strfiles, &user_files, LEN);
if (user_outfile.head && strlen(user_outfile.head->val) != 0)
snprintf(strgoutfile, LEN, "%s", user_outfile.head->val);
else
snprintf(strgoutfile, LEN, "a.out");
if (verbose)
fprintf(stderr, "====> Outputing Makefile\n");
if ((fp = fopen("Makefile", "w")) == NULL)
fprintf(stderr, "Error: could not generate Makefile\n");
else
{
fprintf(fp, "# Makefile generated by %s\n", PACKAGE_NAME);
while (fgets(line, sizeof line, infp) != NULL)
{
replace_variable(line, "@OMPICC@", compilerPath);
replace_variable(line, "@OMPI_INPUT@", strfiles);
replace_variable(line, "@OMPI_OUTPUT@", strgoutfile);
replace_variable(line, "@OMPI_ORTLIB@", ortlibname);
fputs(line, fp);  
}
fclose(fp);
}
}
void ompicc_makefile_compile()
{
char cmd[LEN];
int res;
snprintf(cmd, LEN, "make compile");  
if (verbose)
fprintf(stderr, "====> Running target compile on generated Makefile\n");
if ((res = sysexec(cmd, NULL)) != 0)
_exit(res);
}
void ompicc_makefile_link()
{
char cmd[LEN];
int res;
snprintf(cmd, LEN, "make link");  
if (verbose)
fprintf(stderr, "====> Running target link on generated Makefile\n");
if ((res = sysexec(cmd, NULL)) != 0)
_exit(res);
}
void ompicc_get_envvars()
{
char *t;
if ((t = getenv("OMPI_CPP")) == NULL)
strncpy(PREPROCESSOR, CPPcmd, 511);
else
strncpy(PREPROCESSOR, t, 511);
if ((t = getenv("OMPI_CPPFLAGS")) == NULL)
strncpy(CPPFLAGS, PreprocFlags, 511);
else
strncpy(CPPFLAGS, t, 511);
if ((t = getenv("OMPI_CC")) == NULL)
strncpy(COMPILER, CCcmd, 511);
else
strncpy(COMPILER, t, 511);
if ((t = getenv("OMPI_CFLAGS")) == NULL)
strncpy(CFLAGS, CompileFlags, 511);
else
strncpy(CFLAGS, t, 511);
if ((t = getenv("OMPI_LDFLAGS")) == NULL)
strncpy(LDFLAGS, LinkFlags, 511);
else
strncpy(LDFLAGS, t, 511);
}
void get_ort_flags()
{
char confpath[PATHSIZE];
FILE *fp;
void setflag(char *, char *, void *);
snprintf(confpath, PATHSIZE - 1, "%s/%s/ortconf.%s", LibDir, ortlibname,
ortlibname);
if ((fp = fopen(confpath, "r")) == NULL)
ompicc_error(1, "library `%s' cannot be found\n  (%s is missing)\n",
ortlibname, confpath);
keyval_read(fp, setflag, NULL);
fclose(fp);
}
void setflag(char *key, char *value, void *ignore)
{
if (strcmp(key, "ORTINFO") == 0 && strlen(value) + strlen(ORTINFO) < FLAGSIZE)
strcat(*ORTINFO ? strcat(ORTINFO, " ") : ORTINFO, value);
if (strcmp(key, "CPPFLAGS") == 0 && strlen(value) + strlen(CPPFLAGS) < FLAGSIZE)
strcat(strcat(CPPFLAGS, " "), value);
if (strcmp(key, "CFLAGS") == 0 && strlen(value) + strlen(CFLAGS) < FLAGSIZE)
strcat(strcat(CFLAGS, " "), value);
if (strcmp(key, "LDFLAGS") == 0 && strlen(value) + strlen(LDFLAGS) < FLAGSIZE)
strcat(strcat(LDFLAGS, " "), value);
if (strcmp(key, "CC") == 0 && strlen(COMPILER) < FLAGSIZE)
strcpy(COMPILER, value);
if (strcmp(key, "CPP") == 0 && strlen(PREPROCESSOR) < FLAGSIZE)
strcpy(PREPROCESSOR, value);
}
void get_path(char *argv0, char *path)
{
int i;
memset(path, '\0', PATHSIZE);
for (i = strlen(argv0); i >= 0; i--)
{
if (argv0[i] == '/')
{
strncpy(path, argv0, i + 1);
path[i + 1] = '\0';
break;
}
}
}
int main(int argc, char **argv)
{
arg_t *p;
#if defined(HAVE_REALPATH)
char  argv0[PATHSIZE];
char  path[PATHSIZE];
char  *res;
strcpy(argv0, "");
res = realpath(argv[0], argv0);
if (res == NULL)
strcpy(RealOmpiName, OmpiName);
else
{
get_path(argv0, path);         
strcpy(RealOmpiName, path);
strcat(RealOmpiName, OmpiName);
}
#else
strcpy(RealOmpiName, OmpiName);
#endif
#ifdef PORTABLE_BUILD
char buffer[PATHSIZE];
ssize_t len = readlink("/proc/self/exe", buffer, PATHSIZE - 1);
if (len == -1)
ompicc_error(1, "couldn't retrieve installation path using readlink.\n");
else
if (len == PATHSIZE - 1)
ompicc_error(1, "path to %s too long.\n", PACKAGE_TARNAME);
else
buffer[len] = '\0';
get_path(buffer, InstallPath);
if (strcmp(InstallPath + strlen(InstallPath) - 4, "bin/"))
ompicc_error(1, "invalid installation path for a portable build.\n");
InstallPath[strlen(InstallPath) - 4] = 0;
if (strlen(InstallPath) + 8 + strlen(PACKAGE_TARNAME) + 1 > PATHSIZE)
ompicc_error(1, "path to %s too long.\n", PACKAGE_TARNAME);
strcpy(LibDir, InstallPath);
strcat(LibDir, "lib/");
strcat(LibDir, PACKAGE_TARNAME);
strcpy(IncludeDir, InstallPath);
strcat(IncludeDir, "include/");
strcat(IncludeDir, PACKAGE_TARNAME);
#endif
ompicc_get_envvars();
parse_args(argc, argv);
get_ort_flags();
modules_employ(reqmodules);
if (argc == 1)
{
ompi_info();
fprintf(stderr,
"\nUsage: %s [ompi options] [system compiler options] programfile(s)\n",
argv[0]);
fprintf(stderr,
"\n"
"   Useful OMPi options:\n"
"                  -k: keep intermediate file\n"
"                  -v: be verbose (show the actual steps)\n"
"        --ort=<name>: use a specific OMPi runtime library\n"
"    --devs=<devices>: target the given devices\n"
"           --devinfo: show short info about configured devices\n"
"           --options: show all OMPi options\n"
"\n"
"Use environmental variables OMPI_CPP, OMPI_CC, OMPI_CPPFLAGS,\n"
"OMPI_CFLAGS, OMPI_LDFLAGS to have OMPi use a particular base\n"
"preprocessor and compiler, along with specific flags.\n");
exit(0);
}
if (verbose)
{
ompi_info();
fprintf(stderr, "----\n");
}
if (showdevinfo)
{
fprintf(stderr, "%d configured device module(s): %s\n\n",
nmodules, MODULES_CONFIG);
modules_show_info(longdevinfo);
if (user_files.head == NULL)
exit(0);
}
if (user_files.head == NULL)
{
fprintf(stderr, "No input file specified; "
"run %s with no arguments for help.\n", argv[0]);
exit(0);
}
if (makefile)
ompicc_makefile(argv[0]);
if (makefile)
{
ompicc_makefile_compile();
if (mustlink)
ompicc_makefile_link();
}
else
{
for (p = user_files.head; p != NULL; p = p->next)
ompicc_compile(p->val);
if (mustlink)
ompicc_link();
}
return (0);
}
