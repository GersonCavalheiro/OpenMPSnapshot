


#ifndef commonH
#define commonH

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#endif



#ifndef wclientH
#define wclientH

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include "../common/utility.h"

#define BUFSIZE 1024

#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

struct xorArgs {
unsigned int seed;
char *path;
};

int initDir();
int sendLSTF();
int sendLSTR();
int sendENCR(unsigned int seed, char *path);
int sendDECR(unsigned int seed, char *path);

DWORD WINAPI wthread_LSTF(LPVOID arg);
DWORD WINAPI wthread_LSTR(LPVOID arg);
DWORD WINAPI wthread_ENCR(LPVOID arg);
DWORD WINAPI wthread_DECR(LPVOID arg);

#endif

