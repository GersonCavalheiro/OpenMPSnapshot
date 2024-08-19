


#ifndef commonH
#define commonH

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#endif


#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <math.h>
#include <omp.h>

#include "../common/utility.h"

#define PATHLEN 300
#define BUFSIZE 1024

#define STATE_OK "200"
#define STATE_PENDING "300"
#define STATE_ERROR "400"
#define STATE_UNAVAIL "500"

#define LOGFILE "misc/server.log"

#pragma comment (lib, "Ws2_32.lib")

char folder[PATHLEN];
unsigned short port;
int threadNum;

struct request {
int ID;
char *folder;
char *address;
char *message;
SOCKET sock;
struct request *next;
};

char lastError[600];

int nextReqID;
int numReqs;
struct request *first;
struct request *last;

SRWLOCK reqMutex;
CONDITION_VARIABLE reqCond;

volatile bool run;

int XOR(int a, int b);

int fileXOR(char srcfile[], char dstfile[], long long dim, int seed);

int sendMessage(SOCKET sock, char message[]);

int encrypt(char src[], int seed, SOCKET sock);

int decrypt(char src[], int seed, SOCKET sock);

int listFolder(char folder[], SOCKET sock);

int listRecursive(char folder[], SOCKET sock);

int parseRequest(char folder[], char message[], SOCKET sock);

int addRequest(SRWLOCK *mutex, CONDITION_VARIABLE *cond, char *folder, char *address, char *message, SOCKET sock);

struct request* removeRequest(SRWLOCK *mutex);

DWORD WINAPI task(void *arg);

int executeServer(char folder[], unsigned short port, int threadNum);

void showHelp(char *command);

int main(int argc, char *argv[]);
