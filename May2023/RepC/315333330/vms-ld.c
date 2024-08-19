#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include "libiberty.h"
#include <safe-ctype.h>
#include <sys/stat.h>
#define LNM__STRING 2
#define LNM_C_NAMLENGTH 255
#define PSL_C_SUPER 2
#define PSL_C_USER 3
static int ld_nocall_debug = 0;
static int ld_mkthreads = 0;
static int ld_upcalls = 0;
static int verbose = 0;
static int save_temps = 0;
static int inhibit_exec = 1;
static int debug = 0;
static int staticp = 1;
static int share = 0;
static int link_cmd_maxlen = 0;
static char *link_cmd = 0;
static int link_cmd_len = 0;
static char *sharebasename;
static const char *exefullfilename;
static const char *exefilename;
static const char **search_dirs;
static int search_dirs_len;
static void addarg (const char *);
static int is_regular_file (char *);
static char *to_host_file_spec (char *);
static char *locate_lib (char *);
static const char *expand_lib (char *);
static void preprocess_args (int, char **);
static void process_args (int, char **);
static void maybe_set_link_compat (void);
static int set_exe (const char *);
#ifdef VMS
static int translate_unix (char *, int);
#endif

static void
addarg (const char *str)
{
int l = strlen (str);
if (link_cmd_len + l >= link_cmd_maxlen)
{
link_cmd_maxlen = link_cmd_len + l + 1024;
link_cmd = XRESIZEVEC (char, link_cmd, link_cmd_maxlen);
}
memcpy (link_cmd + link_cmd_len, str, l);
link_cmd_len += l;
}
static int
is_regular_file (char *name)
{
int ret;
struct stat statbuf;
ret = stat (name, &statbuf);
return !ret && S_ISREG (statbuf.st_mode);
}
#ifdef VMS
static char new_host_filespec [255];
static char filename_buff [256];
static int
translate_unix (char *name, int type ATTRIBUTE_UNUSED)
{
strcpy (filename_buff, name);
return 0;
}
#endif
static char *
to_host_file_spec (char *filespec)
{
#ifdef VMS
if (strchr (filespec, ']') || strchr (filespec, ':'))
{
return filespec;
}
else
{
strcpy (filename_buff, filespec);
decc$to_vms (filespec, translate_unix, 1, 1);
strcpy (new_host_filespec, filename_buff);
return new_host_filespec;
}
#else
return filespec;
#endif
}
static char *
locate_lib (char *lib_name)
{
int lib_len = strlen (lib_name);
const char *exts[3];
int i;
if (staticp)
{
exts[0] = ".a";
exts[1] = ".olb";
exts[2] = ".exe";
}
else
{
exts[0] = ".exe";
exts[1] = ".a";
exts[2] = ".olb";
}
for (i = 0; i < search_dirs_len; i++)
{
char *buf;
int l;
int j;
l = strlen (search_dirs[i]);
buf = (char *)alloca (l + 4 + lib_len + 4 + 1);
memcpy (buf, search_dirs[i], l);
memcpy (buf + l, "/lib", 4);
l += 4;
memcpy (buf + l, lib_name, lib_len);
l += lib_len;
for (j = 0; j < 3; j++)
{
strcpy (buf + l, exts[j]);
if (is_regular_file (buf))
return xstrdup (to_host_file_spec (buf));
}
}
return NULL;
}
static const char *
expand_lib (char *name)
{
char *lib_path;
if (strcmp (name, "c") == 0)
return NULL;
if (strcmp (name, "m") == 0)
return NULL;
lib_path = locate_lib (name);
if (lib_path)
return lib_path;
fprintf (stderr,
"Couldn't locate library: lib%s.exe, lib%s.a or lib%s.olb\n",
name, name, name);
exit (EXIT_FAILURE);
}
static void
preprocess_args (int argc, char **argv)
{
int i;
for (i = 1; i < argc; i++)
if (strcmp (argv[i], "-shared") == 0)
{
share = 1;
break;
}
for (i = 1; i < argc; i++)
if (strcmp (argv[i], "-o") == 0)
{
int len;
i++;
exefilename = lbasename (argv[i]);
exefullfilename = xstrdup (to_host_file_spec (argv[i]));
if (share)
addarg(" /share=");
else
addarg (" /exe=");
addarg (exefullfilename);
if (share)
{
char *ptr;
ptr = strchr (argv[i], ']');
if (ptr == NULL)
ptr = strchr (argv[i], ':');
if (ptr == NULL)
ptr = strchr (argv[i], '/');
if (ptr == NULL)
sharebasename = xstrdup (argv[i]);
else
sharebasename = xstrdup (ptr + 1);
len = strlen (sharebasename);
if (strncasecmp (&sharebasename[len-4], ".exe", 4) == 0)
sharebasename[len - 4] = 0;
for (ptr = sharebasename; *ptr; ptr++)
*ptr = TOUPPER (*ptr);
}
}
if (exefullfilename == NULL && !share)
{
exefilename = "a_out.exe";
exefullfilename = "a_out.exe";
addarg (xstrdup (" /exe=a_out.exe"));
}
}
static void
process_args (int argc, char **argv)
{
int i;
for (i = 1; i < argc; i++)
{
if (strncmp (argv[i], "-L", 2) == 0)
{
search_dirs = XRESIZEVEC(const char *, search_dirs,
search_dirs_len + 1);
search_dirs[search_dirs_len++] = &argv[i][2];
}
else if (strcmp (argv[i], "-v") == 0)
verbose++;
else if (strcmp (argv[i], "--version") == 0)
{
fprintf (stdout, "VMS Linker\n");
exit (EXIT_SUCCESS);
}
else if (strcmp (argv[i], "--help") == 0)
{
fprintf (stdout, "VMS Linker\n");
exit (EXIT_SUCCESS);
}
else if (strcmp (argv[i], "-g0") == 0)
addarg ("/notraceback");
else if (strncmp (argv[i], "-g", 2) == 0)
{
addarg ("/debug");
debug = 1;
}
else if (strcmp (argv[i], "-static") == 0)
staticp = 1;
else if (strcmp (argv[i], "-map") == 0)
{
char *buff, *ptr;
buff = (char *) xstrdup (exefullfilename);
ptr = strrchr (buff, '.');
if (ptr)
*ptr = 0;
strcat (buff, ".map");
addarg ("/map=");
addarg (buff);
addarg (".map");
addarg ("/full");
free (buff);
}
else if (strcmp (argv[i], "-save-temps") == 0)
save_temps = 1;
else if (strcmp (argv[i], "--noinhibit-exec") == 0)
inhibit_exec = 0;
}
}
#ifdef VMS
typedef struct dsc
{
unsigned short len, mbz;
const char *adr;
} Descriptor;
struct lst
{
unsigned short buflen, item_code;
const void *bufaddr;
void *retlenaddr;
};
static struct
{
struct lst items [1];
unsigned int terminator;
} item_lst1;
static struct
{
struct lst items [2];
unsigned int terminator;
} item_lst2;
static void
maybe_set_link_compat (void)
{
char lnm_buff [LNM_C_NAMLENGTH];
unsigned int lnm_buff_len;
int status;
Descriptor tabledsc, linkdsc;
tabledsc.adr = "LNM$JOB";
tabledsc.len = strlen (tabledsc.adr);
tabledsc.mbz = 0;
linkdsc.adr = "GCC_LD_SYS$LIBRARY";
linkdsc.len = strlen (linkdsc.adr);
linkdsc.mbz = 0;
item_lst1.items[0].buflen = LNM_C_NAMLENGTH;
item_lst1.items[0].item_code = LNM__STRING;
item_lst1.items[0].bufaddr = lnm_buff;
item_lst1.items[0].retlenaddr = &lnm_buff_len;
item_lst1.terminator = 0;
status = SYS$TRNLNM
(0,          
&tabledsc,  
&linkdsc,   
0,          
&item_lst1);
if ((status & 1) == 1)
{
unsigned char acmode = PSL_C_USER; 
const char *syslib = "SYS$SYSROOT:[SYSLIB]"; 
tabledsc.adr = "LNM$PROCESS";
tabledsc.len = strlen (tabledsc.adr);
tabledsc.mbz = 0;
linkdsc.adr = "SYS$LIBRARY";
linkdsc.len = strlen (linkdsc.adr);
linkdsc.mbz = 0;
item_lst2.items[0].buflen = lnm_buff_len;
item_lst2.items[0].item_code = LNM__STRING;
item_lst2.items[0].bufaddr = lnm_buff;
item_lst2.items[0].retlenaddr = 0;
item_lst2.items[1].buflen = strlen (syslib);
item_lst2.items[1].item_code = LNM__STRING;
item_lst2.items[1].bufaddr = syslib;
item_lst2.items[1].retlenaddr = 0;
item_lst2.terminator = 0;
status = SYS$CRELNM
(0,          
&tabledsc,  
&linkdsc,   
&acmode,    
&item_lst2);
}
tabledsc.adr = "LNM$JOB";
tabledsc.len = strlen (tabledsc.adr);
tabledsc.mbz = 0;
linkdsc.adr = "GCC_LD_LINK";
linkdsc.len = strlen (linkdsc.adr);
linkdsc.mbz = 0;
item_lst1.items[0].buflen = LNM_C_NAMLENGTH;
item_lst1.items[0].item_code = LNM__STRING;
item_lst1.items[0].bufaddr = lnm_buff;
item_lst1.items[0].retlenaddr = &lnm_buff_len;
item_lst1.terminator = 0;
status = SYS$TRNLNM
(0,          
&tabledsc,  
&linkdsc,   
0,          
&item_lst1);
if ((status & 1) == 1)
{
unsigned char acmode = PSL_C_USER; 
tabledsc.adr = "LNM$PROCESS";
tabledsc.len = strlen (tabledsc.adr);
tabledsc.mbz = 0;
linkdsc.adr = "LINK";
linkdsc.len = strlen (linkdsc.adr);
linkdsc.mbz = 0;
item_lst1.items[0].buflen = lnm_buff_len;
item_lst1.items[0].item_code = LNM__STRING;
item_lst1.items[0].bufaddr = lnm_buff;
item_lst1.items[0].retlenaddr = 0;
item_lst1.terminator = 0;
status = SYS$CRELNM
(0,          
&tabledsc,  
&linkdsc,   
&acmode,    
&item_lst1);
}
}
#else
static void
maybe_set_link_compat (void)
{
}
#endif
static int
set_exe (const char *arg)
{
char allargs [1024];
int res;
snprintf (allargs, sizeof (allargs),
"$@gnu:[bin]set_exe %s %s", exefullfilename, arg);
if (verbose)
printf ("%s\n", allargs);
res = system (allargs);
if (verbose > 1)
printf ("$!status = %d\n", res);
if ((res & 1) != 1)
{
fprintf (stderr, "ld error: popen set_exe\n");
return 1;
}
return 0;
}
int
main (int argc, char **argv)
{
char *vmsdwarf2spec = 0;
char *vmsdwarf2ehspec = 0;
int i;
char cwdev[128], *devptr;
int cwdevlen;
FILE *optfile;
char *cwd, *ptr;
char *optfilename;
int status = 0;
if (getenv ("GNAT$LD_NOCALL_DEBUG"))
ld_nocall_debug = 1;
if (getenv ("GNAT$LD_MKTHREADS"))
ld_mkthreads = 1;
if (getenv ("GNAT$LD_UPCALLS"))
ld_upcalls = 1;
if (getenv ("GNAT$LD_SHARED_LIBS"))
staticp = 0;
#ifdef VMS
cwd = getcwd (0, 1024, 1);
#else
cwd = getcwd (0, 1024);
strcat (cwd, "/");
#endif
devptr = strchr (cwd, ':');
if (devptr)
cwdevlen = (devptr - cwd) + 1;
else
cwdevlen = 0;
memcpy (cwdev, cwd, cwdevlen);
cwdev [cwdevlen] = '\0';
maybe_set_link_compat ();
addarg ("$ link");
preprocess_args (argc , argv);
process_args (argc , argv);
if (!verbose)
addarg ("/noinform");
optfilename = (char *) xmalloc (strlen (exefilename) + 13);
strcpy (optfilename, exefilename);
ptr = strrchr (optfilename, '.');
if (ptr)
*ptr = 0;
strcat (optfilename, ".opt_tmpfile");
optfile = fopen (optfilename, "w");
for (i = 1; i < argc; i++)
{
int arg_len = strlen (argv[i]);
if (arg_len > 6 && strncasecmp (argv[i], "IDENT=", 6) == 0)
{
break;
}
else if (arg_len > 17
&& strncasecmp (argv[i], "--identification=", 17) == 0)
{
fprintf (optfile, "case_sensitive=yes\n");
fprintf (optfile, "IDENTIFICATION=\"%-.15s\"\n", &argv[i][17]);
fprintf (optfile, "case_sensitive=NO\n");
}
}
for (i = 1; i < argc; i++)
{
int arg_len = strlen (argv[i]);
if (strcmp (argv[i], "-o") == 0)
{
i++;
}
else if (arg_len > 2 && strncmp (argv[i], "-l", 2) == 0)
{
const char *libname;
libname = expand_lib (&argv[i][2]);
if (libname != NULL)
{
int len = strlen (libname);
const char *ext;
if (len > 4 && strcasecmp (&libname [len-4], ".exe") == 0)
ext = "/shareable";
else
ext = "/library";
if (libname[0] == '[')
fprintf (optfile, "%s%s%s\n", cwdev, libname, ext);
else
fprintf (optfile, "%s%s\n", libname, ext);
}
}
else if (strcmp (argv[i], "-v" ) == 0
|| strncmp (argv[i], "-g", 2 ) == 0
|| strcmp (argv[i], "-static" ) == 0
|| strcmp (argv[i], "-map" ) == 0
|| strcmp (argv[i], "-save-temps") == 0
|| strcmp (argv[i], "--noinhibit-exec") == 0
|| (arg_len > 2 && strncmp (argv[i], "-L", 2) == 0)
|| (arg_len >= 6 && strncmp (argv[i], "-share", 6) == 0))
{
}
else if (strncmp (argv[i], "--opt=", 6) == 0)
fprintf (optfile, "%s\n", argv[i] + 6);
else if (arg_len > 1 && argv[i][0] == '@')
{
FILE *atfile;
char *ptr, *ptr1;
struct stat statbuf;
char *buff;
int len;
if (stat (&argv[i][1], &statbuf))
{
fprintf (stderr, "Couldn't open linker response file: %s\n",
&argv[i][1]);
exit (EXIT_FAILURE);
}
buff = (char *) xmalloc (statbuf.st_size + 1);
atfile = fopen (&argv[i][1], "r");
fgets (buff, statbuf.st_size + 1, atfile);
fclose (atfile);
len = strlen (buff);
if (buff [len - 1] == '\n')
{
buff [len - 1] = 0;
len--;
}
ptr = buff;
do
{
ptr1 = strchr (ptr, ' ');
if (ptr1)
*ptr1 = 0;
ptr = to_host_file_spec (ptr);
if (ptr[0] == '[')
fprintf (optfile, "%s%s\n", cwdev, ptr);
else
fprintf (optfile, "%s\n", ptr);
ptr = ptr1 + 1;
}
while (ptr1);
}
else if ((argv[i][0] == '/') && (strchr (&argv[i][1], '/') == 0))
{
addarg (argv[i]);
}
else if (arg_len > 4
&& strncasecmp (&argv[i][arg_len-4], ".opt", 4) == 0)
{
FILE *optfile1;
char buff[256];
optfile1 = (fopen) (argv[i], "r");
if (optfile1 == 0)
{
perror (argv[i]);
status = 1;
goto cleanup_and_exit;
}
while (fgets (buff, sizeof (buff), optfile1))
fputs (buff, optfile);
fclose (optfile1);
}
else if (arg_len > 7 && strncasecmp (argv[i], "GSMATCH", 7) == 0)
fprintf (optfile, "%s\n", argv[i]);
else if (arg_len > 6 && strncasecmp (argv[i], "IDENT=", 6) == 0)
{
fprintf (optfile, "case_sensitive=yes\n");
fprintf (optfile, "IDENT=\"%15.15s\"\n", &argv[i][6]);
fprintf (optfile, "case_sensitive=NO\n");
}
else if (arg_len > 17
&& strncasecmp (argv[i], "--identification=", 17) == 0)
{
}
else
{
const char *file;
const char *addswitch = NULL;
char *buff;
int buff_len;
int is_cld = 0;
file = to_host_file_spec (argv[i]);
arg_len = strlen (file);
if (arg_len > 4 && strcasecmp (&file[arg_len - 4], ".exe") == 0)
addswitch = "/shareable";
else if (arg_len > 4 && strcasecmp (&file[arg_len - 4], ".cld") == 0)
{
addswitch = "/shareable";
is_cld = 1;
}
else if (arg_len > 2 && strcasecmp (&file[arg_len - 2], ".a") == 0)
addswitch = "/lib";
else if (arg_len > 4 && strcasecmp (&file[arg_len - 4], ".olb") == 0)
addswitch = "/lib";
if (file[0] == '[')
{
buff = (char *) xmalloc (cwdevlen + arg_len + 1);
sprintf (buff, "%s%s", cwdev, file);
}
else if (strchr (file, ':'))
{
buff = xstrdup (file);
}
else
{
buff = (char *) xmalloc (strlen (cwd) + arg_len + 1);
sprintf (buff, "%s%s", cwd, file);
}
buff_len = strlen (buff);
if (buff_len >= 15
&& strcasecmp (&buff[buff_len - 14], "vms-dwarf2eh.o") == 0)
{
vmsdwarf2ehspec = xstrdup (buff);
}
else if (buff_len >= 13
&& strcasecmp (&buff[buff_len - 12], "vms-dwarf2.o") == 0)
{
vmsdwarf2spec = xstrdup (buff);
}
else if (is_cld)
{
addarg (buff);
addarg (addswitch);
addarg (",");
}
else
{
fprintf (optfile, "%s%s\n",
buff, addswitch != NULL ? addswitch : "");
}
free (buff);
}
}
if (vmsdwarf2ehspec)
{
fprintf (optfile, "case_sensitive=yes\n");
fprintf (optfile, "cluster=DWARF2eh,,,%s\n", vmsdwarf2ehspec);
fprintf (optfile, "collect=DWARF2eh,eh_frame\n");
fprintf (optfile, "case_sensitive=NO\n");
}
if (debug && vmsdwarf2spec)
{
fprintf (optfile, "case_sensitive=yes\n");
fprintf (optfile, "cluster=DWARF2debug,,,%s\n", vmsdwarf2spec);
fprintf (optfile, "collect=DWARF2debug,debug_abbrev,debug_aranges,-\n");
fprintf (optfile, " debug_frame,debug_info,debug_line,debug_loc,-\n");
fprintf (optfile, " debug_macinfo,debug_pubnames,debug_str,-\n");
fprintf (optfile, " debug_zzzzzz\n");
fprintf (optfile, "case_sensitive=NO\n");
}
if (debug && share && vmsdwarf2spec)
{
fprintf (optfile, "case_sensitive=yes\n");
fprintf (optfile, "symbol_vector=(-\n");
fprintf (optfile,
"%s$DWARF2.DEBUG_ABBREV/$dwarf2.debug_abbrev=DATA,-\n",
sharebasename);
fprintf (optfile,
"%s$DWARF2.DEBUG_ARANGES/$dwarf2.debug_aranges=DATA,-\n",
sharebasename);
fprintf (optfile, "%s$DWARF2.DEBUG_FRAME/$dwarf2.debug_frame=DATA,-\n",
sharebasename);
fprintf (optfile, "%s$DWARF2.DEBUG_INFO/$dwarf2.debug_info=DATA,-\n",
sharebasename);
fprintf (optfile, "%s$DWARF2.DEBUG_LINE/$dwarf2.debug_line=DATA,-\n",
sharebasename);
fprintf (optfile, "%s$DWARF2.DEBUG_LOC/$dwarf2.debug_loc=DATA,-\n",
sharebasename);
fprintf (optfile,
"%s$DWARF2.DEBUG_MACINFO/$dwarf2.debug_macinfo=DATA,-\n",
sharebasename);
fprintf (optfile,
"%s$DWARF2.DEBUG_PUBNAMES/$dwarf2.debug_pubnames=DATA,-\n",
sharebasename);
fprintf (optfile, "%s$DWARF2.DEBUG_STR/$dwarf2.debug_str=DATA,-\n",
sharebasename);
fprintf (optfile, "%s$DWARF2.DEBUG_ZZZZZZ/$dwarf2.debug_zzzzzz=DATA)\n",
sharebasename);
fprintf (optfile, "case_sensitive=NO\n");
}
fprintf (optfile, "PSECT_ATTR=LIB$INITIALIZE,GBL\n");
fclose (optfile);
addarg (" ");
addarg (optfilename);
addarg ("/opt");
if (verbose)
printf ("%s\n", link_cmd);
status = system (link_cmd);
if (verbose > 1)
printf ("$!status = %d\n", status);
if ((status & 1) != 1)
{
status = 1;
goto cleanup_and_exit;
}
if (debug && !share && ld_nocall_debug)
{
status = set_exe ("/flags=nocall_debug");
if (status != 0)
goto cleanup_and_exit;
}
if (!share && ld_mkthreads)
{
status = set_exe ("/flags=mkthreads");
if (status != 0)
goto cleanup_and_exit;
}
if (!share && ld_upcalls)
{
status = set_exe ("/flags=upcalls");
if (status != 0)
goto cleanup_and_exit;
}
status = 0;
cleanup_and_exit:
if (!save_temps)
remove (optfilename);
if (status == 0)
exit (EXIT_SUCCESS);
if (exefullfilename && inhibit_exec == 1)
remove (exefullfilename);
exit (EXIT_FAILURE);
}
