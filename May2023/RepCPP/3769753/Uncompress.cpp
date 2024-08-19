

#include "config.h"
#if HAVE_LIBDL

#include "Fcontrol.h"
#include "SignalHandler.h"
#include "StringUtil.h"
#include <cassert>
#include <cstdio> 
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <unistd.h>

using namespace std;

static const char* wgetExec(const string& path)
{
return
startsWith(path, "http:
startsWith(path, "https:
startsWith(path, "ftp:
NULL;
}

static const char* zcatExec(const string& path)
{
return
endsWith(path, ".ar") ? "ar -p" :
endsWith(path, ".tar") ? "tar -xOf" :
endsWith(path, ".tar.Z") ? "tar -zxOf" :
endsWith(path, ".tar.gz") ? "tar -zxOf" :
endsWith(path, ".tar.bz2") ? "tar -jxOf" :
endsWith(path, ".tar.xz") ?
"tar --use-compress-program=xzdec -xOf" :
endsWith(path, ".Z") ? "gunzip -c" :
endsWith(path, ".gz") ? "gunzip -c" :
endsWith(path, ".bz2") ? "bunzip2 -c" :
endsWith(path, ".xz") ? "xzdec -c" :
endsWith(path, ".zip") ? "unzip -p" :
endsWith(path, ".bam") ? "samtools view -h" :
endsWith(path, ".cram") ? "samtools view -h" :
endsWith(path, ".jf") ? "jellyfish dump" :
endsWith(path, ".jfq") ? "jellyfish qdump" :
endsWith(path, ".sra") ? "fastq-dump -Z --split-spot" :
endsWith(path, ".url") ? "wget -O- -i" :
endsWith(path, ".fqz") ? "fqz_comp -d" :
NULL;
}

extern "C" {


static int uncompress(const char *path)
{
const char *wget = wgetExec(path);
const char *zcat = wget != NULL ? wget : zcatExec(path);
assert(zcat != NULL);

int fd[2];
if (pipe(fd) == -1)
return -1;
int err = setCloexec(fd[0]);
assert(err == 0);
(void)err;

char arg0[16], arg1[16], arg2[16];
int n = sscanf(zcat, "%s %s %s", arg0, arg1, arg2);
assert(n == 2 || n == 3);


#if HAVE_WORKING_VFORK
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
pid_t pid = vfork();
#pragma GCC diagnostic pop
#else
pid_t pid = fork();
#endif
if (pid == -1)
return -1;

if (pid == 0) {
dup2(fd[1], STDOUT_FILENO);
close(fd[1]);
if (n == 2)
execlp(arg0, arg0, arg1, path, NULL);
else
execlp(arg0, arg0, arg1, arg2, path, NULL);
perror(arg0);
_exit(EXIT_FAILURE);
} else {
close(fd[1]);
return fd[0];
}
}


static FILE* funcompress(const char* path)
{
int fd = uncompress(path);
if (fd == -1) {
perror(path);
exit(EXIT_FAILURE);
}
return fdopen(fd, "r");
}

typedef FILE* (*fopen_t)(const char *path, const char *mode);


FILE *fopen(const char *path, const char *mode)
{
static fopen_t real_fopen;
if (real_fopen == NULL)
real_fopen = (fopen_t)dlsym(RTLD_NEXT, "fopen");
if (real_fopen == NULL) {
fprintf(stderr, "error: dlsym fopen: %s\n", dlerror());
exit(EXIT_FAILURE);
}

if (wgetExec(path) != NULL)
return funcompress(path);

FILE* stream = real_fopen(path, mode);
if (string(mode) != "r" || !stream || zcatExec(path) == NULL)
return stream;
else {
fclose(stream);
return funcompress(path);
}
}


FILE *fopen64(const char *path, const char *mode)
{
static fopen_t real_fopen64;
if (real_fopen64 == NULL)
real_fopen64 = (fopen_t)dlsym(RTLD_NEXT, "fopen64");
if (real_fopen64 == NULL) {
fprintf(stderr, "error: dlsym fopen64: %s\n", dlerror());
exit(EXIT_FAILURE);
}

if (wgetExec(path) != NULL)
return funcompress(path);

FILE* stream = real_fopen64(path, mode);
if (string(mode) != "r" || !stream || zcatExec(path) == NULL)
return stream;
else {
fclose(stream);
return funcompress(path);
}
}

typedef int (*open_t)(const char *path, int flags, mode_t mode);


int open(const char *path, int flags, mode_t mode)
{
static open_t real_open;
if (real_open == NULL)
real_open = (open_t)dlsym(RTLD_NEXT, "open");
if (real_open == NULL) {
fprintf(stderr, "error: dlsym open: %s\n", dlerror());
exit(EXIT_FAILURE);
}

if (wgetExec(path) != NULL)
return uncompress(path);

int filedesc = real_open(path, flags, mode);
if (mode != ios_base::in || filedesc < 0
|| zcatExec(path) == NULL)
return filedesc;
else {
close(filedesc);
return uncompress(path);
}
}

} 

#endif 


bool uncompress_init()
{
#if HAVE_LIBDL
signalInit();
#endif
return HAVE_LIBDL;
}
