#define _REENTRANT
#define _THREAD_SAFE
#ifdef MaRTE
#pragma weak __get_errno
#pragma weak __set_errno
#endif
#include <errno.h>
int
__get_errno(void)
{
return errno;
}
#ifndef __ANDROID__
void
__set_errno(int err)
{
errno = err;
}
#endif
