

#include "wserver.h"

int XOR(int a, int b) {
return a^b;
}

int fileXOR(char srcfile[], char dstfile[], long long dim, int seed) {
HANDLE src = CreateFile(srcfile,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS,NULL);
if (src==INVALID_HANDLE_VALUE) {
sprintf(lastError, "Errore apertura file %s.\n", srcfile);
return 400;
}

HANDLE dst = CreateFile(dstfile, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_FLAG_RANDOM_ACCESS, NULL);
if (dst==INVALID_HANDLE_VALUE) {
sprintf(lastError, "Errore apertura file %s.\n", dstfile);
CloseHandle(src);
return 400;
}

OVERLAPPED srcoverlap;
memset(&srcoverlap,0,sizeof(srcoverlap));
if (!LockFileEx(src, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 0, dim, &srcoverlap)) {
sprintf(lastError, "Errore lock su file %s.\n", srcfile);
CloseHandle(src);
CloseHandle(dst);

return 500;
}

OVERLAPPED dstoverlap;
memset(&dstoverlap, 0, sizeof(dstoverlap));
if (!LockFileEx(dst, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 0, dim, &dstoverlap)) {
sprintf(lastError, "Errore lock su file %s.\n", dstfile);
UnlockFileEx(src,0,0,dim,&srcoverlap);
CloseHandle(src);
CloseHandle(dst);

return 500;
}

LARGE_INTEGER fileSize, fileMapSize, mapViewSize, fileMapStart;
DWORD granularity;
SYSTEM_INFO sysInfo;
long offset;

GetSystemInfo(&sysInfo);
granularity = sysInfo.dwAllocationGranularity;

LARGE_INTEGER LIrounded;
LIrounded.HighPart = 0;
LIrounded.LowPart = dim - 2;
if(!SetFilePointerEx(dst,LIrounded,NULL,FILE_BEGIN)){
sprintf(lastError, "Errore stretch file %s.\n", dstfile);
UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);

CloseHandle(src);
CloseHandle(dst);

return 500;
}

char buff[] = "\0";
if (!WriteFile(dst, buff, sizeof(buff), NULL, NULL)) {
sprintf(lastError, "Errore scrittura su file %s.\n", dstfile);
UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);

CloseHandle(src);
CloseHandle(dst);

return 500;
}

srand(seed);

HANDLE handle_srcmap = CreateFileMapping(src, NULL, PAGE_READONLY, 0, 0, NULL);
if (handle_srcmap == NULL) {
sprintf(lastError, "Errore file mapping su file %s: %d\n", srcfile, GetLastError());
UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);

CloseHandle(src);
CloseHandle(dst);

return 500;
}

HANDLE handle_dstmap = CreateFileMapping(dst, NULL, PAGE_READWRITE, 0, 0, NULL);
if (handle_dstmap == NULL) {
sprintf(lastError, "Errore file mapping su file %s: %d\n", dstfile, GetLastError());
CloseHandle(handle_srcmap);

UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);

CloseHandle(src);
CloseHandle(dst);

return 500;
}

if (dim <= 256 * 1024) {
MEMORYSTATUSEX memstatus;
memstatus.dwLength = sizeof(memstatus);
GlobalMemoryStatusEx(&memstatus);
long freeMem = memstatus.ullAvailVirtual;
if (freeMem <= 3 * dim) {
sprintf(lastError, "RAM insufficiente per aprire il file %s.\n", srcfile);
CloseHandle(handle_srcmap);
CloseHandle(handle_dstmap);

UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);

CloseHandle(src);
CloseHandle(dst);

return 500;
}

char *srcmap = (char *)MapViewOfFile(handle_srcmap,FILE_MAP_READ,0,0,0);
if ((LPVOID)srcmap == NULL) {
sprintf(lastError, "Errore mapview su file %s: %d.\n", srcfile, GetLastError());
CloseHandle(handle_srcmap);
CloseHandle(handle_dstmap);

UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);

CloseHandle(src);
CloseHandle(dst);

return 500;
}

char *dstmap = (char *)MapViewOfFile(handle_dstmap, FILE_MAP_ALL_ACCESS, 0, 0, 0);
if ((LPVOID)dstmap == NULL) {
sprintf(lastError, "Errore mapview su file %s: %d.\n", dstfile, GetLastError());
UnmapViewOfFile((LPVOID)srcmap);
CloseHandle(handle_srcmap);
CloseHandle(handle_dstmap);

UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);

CloseHandle(src);
CloseHandle(dst);

return 500;
}

long keyDim = (long)ceil((double)dim / 4) * 4;
int *key;
key = malloc(keyDim * sizeof(int));
if (key == NULL) {
sprintf(lastError, "Errore malloc.\n");

UnmapViewOfFile((LPVOID)srcmap);
UnmapViewOfFile((LPVOID)dstmap);
CloseHandle(handle_srcmap);
CloseHandle(handle_dstmap);
UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);
CloseHandle(src);
CloseHandle(dst);

return 500;
}
for (long i = 0; i<keyDim; i++) {
key[i] = rand() % 65536; 
}

long i, j;
for (i = 0, j = 0; i<dim && j<keyDim; i += 4, j++) {
dstmap[i] = (char)(XOR((int)srcmap[i], key[j]));
dstmap[i + 1] = (char)(XOR((int)srcmap[i + 1], key[j]));
dstmap[i + 2] = (char)(XOR((int)srcmap[i + 2], key[j]));
dstmap[i + 3] = (char)(XOR((int)srcmap[i + 3], key[j]));
}

free(key);
UnmapViewOfFile((LPVOID)srcmap);
UnmapViewOfFile((LPVOID)dstmap);
CloseHandle(handle_srcmap);
CloseHandle(handle_dstmap);
UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);
CloseHandle(src);
CloseHandle(dst);
}
else {
long fiveMB = 5 * pow(2, 20);
int chunks = (int)ceil((double)dim / fiveMB);

for (int c = 0; c<chunks; c++) {
MEMORYSTATUSEX memstatus;
memstatus.dwLength = sizeof(memstatus);
GlobalMemoryStatusEx(&memstatus);
long freeMem = memstatus.ullAvailVirtual;
if (freeMem <= 2 * fiveMB) {
sprintf(lastError, "RAM insufficiente per aprire il file %s.\n", srcfile);
CloseHandle(handle_srcmap);
CloseHandle(handle_dstmap);

UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);

CloseHandle(src);
CloseHandle(dst);

return 500;
}

long start = (c)*fiveMB;
long end = (c + 1)*fiveMB;
long realEnd = end;
if (dim<realEnd)
realEnd = dim;

long chunkDim = realEnd - start;
fileMapStart.QuadPart = (start/granularity)*granularity;
offset = start - fileMapStart.QuadPart;
if (dim - fileMapStart.LowPart < chunkDim) 
chunkDim = dim - fileMapStart.LowPart;
mapViewSize.QuadPart = (start%granularity) + chunkDim;

char *srcmap = (char *)MapViewOfFile(handle_srcmap, FILE_MAP_READ, fileMapStart.HighPart, fileMapStart.LowPart, mapViewSize.QuadPart);
if ((LPVOID)srcmap == NULL) {
sprintf(lastError, "Errore mapview su file %s, chunk #%i: %d\n", srcfile, c, GetLastError());
CloseHandle(handle_srcmap);
CloseHandle(handle_dstmap);

UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);

CloseHandle(src);
CloseHandle(dst);

return 500;
}

srcmap += offset;

char *dstmap = (char *)MapViewOfFile(handle_dstmap, FILE_MAP_ALL_ACCESS, fileMapStart.HighPart, fileMapStart.LowPart, mapViewSize.QuadPart);
if ((LPVOID)dstmap == NULL) {
sprintf(lastError, "Errore mapview su file %s: %d\n", dstfile, GetLastError());
UnmapViewOfFile((LPVOID)srcmap);
CloseHandle(handle_srcmap);
CloseHandle(handle_dstmap);

UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);

CloseHandle(src);
CloseHandle(dst);

return 500;
}

dstmap += offset;

int mpThreads = (int)ceil((double)chunkDim / (256 * 1024));

long keyDimT = (long)ceil((double)chunkDim / (mpThreads*4));
int *key;
key = malloc(mpThreads*keyDimT*sizeof(int));
if (key == NULL) {
sprintf(lastError, "Errore malloc.\n");

UnmapViewOfFile((LPVOID)srcmap);
UnmapViewOfFile((LPVOID)dstmap);
CloseHandle(handle_srcmap);
CloseHandle(handle_dstmap);
UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);
CloseHandle(src);
CloseHandle(dst);

return 500;
}
for (long j = 0; j<mpThreads; j++) {
for (long i = 0; i<keyDimT; i++) {
key[j*mpThreads + i] = rand() % 65536; 
}
}

#pragma omp parallel num_threads(mpThreads)
{
int threadID = omp_get_thread_num();
int min = (threadID) * 256 * 1024;
int max = (threadID + 1) * 256 * 1024;

for (long i = min; i<max && i<chunkDim; i += 4) {
int val = key[(threadID*mpThreads) + ((i - min) / 4)];
dstmap[i] = (char)(XOR((int)srcmap[i], val));
dstmap[i + 1] = (char)(XOR((int)srcmap[i + 1], val));
dstmap[i + 2] = (char)(XOR((int)srcmap[i + 2], val));
dstmap[i + 3] = (char)(XOR((int)srcmap[i + 3], val));
}
}

free(key);
UnmapViewOfFile((LPVOID)srcmap);
UnmapViewOfFile((LPVOID)dstmap);
}
CloseHandle(handle_srcmap);
CloseHandle(handle_dstmap);
UnlockFileEx(src, 0, 0, dim, &srcoverlap);
UnlockFileEx(dst, 0, 0, dim, &dstoverlap);
CloseHandle(src);
CloseHandle(dst);
}
return 200;
}

int sendMessage(SOCKET sock, char message[]) {
char buf[BUFSIZE];
ZeroMemory(buf,BUFSIZE);
strncpy(buf,message,BUFSIZE);

int res = send(sock, buf, BUFSIZE, 0);
if (res == SOCKET_ERROR) {
printf("Errore send: %d\n", WSAGetLastError());
closesocket(sock);
return 1;
}

return 0;
}

int encrypt(char src[], int seed, SOCKET sock) {
char dst[PATHLEN] = "";
strncpy(dst, src, strlen(src));
strncat(dst, "_enc", 5);

LARGE_INTEGER dim;
HANDLE srcfile;

srcfile = CreateFile(src, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

if (srcfile == INVALID_HANDLE_VALUE) {
sprintf(lastError, "File %s non esistente.\n", src);
return 400;
}

if (!GetFileSizeEx(srcfile,&dim)) {
sprintf(lastError, "Errore nel calcolo dimensione del file %s.\n", src);
CloseHandle(srcfile);
return 500;
}
CloseHandle(srcfile);

int ret = fileXOR(src, dst, (long long)dim.QuadPart, seed);

if (ret == 200 && !DeleteFile(src)) {
sprintf(lastError, "Errore nella cancellazione del file %s: %d\n", src, GetLastError());
return 500;
}

return ret;
}

int decrypt(char src[], int seed, SOCKET sock) {
char *enc = NULL;
char *temp = strstr(src, "_enc");
while (temp) {
enc = temp++;
temp = strstr(temp, "_enc");
}

if (enc == NULL || strlen(enc) != 4) {
sprintf(lastError, "Il file %s non e' un file cifrato.\n", src);
return 400;
}

char dst[PATHLEN] = "";
strncpy(dst, src, strlen(src) - 4);

LARGE_INTEGER dim;
HANDLE srcfile;

srcfile = CreateFile(src, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
if (srcfile == INVALID_HANDLE_VALUE) {
sprintf(lastError, "File %s non esistente.\n", src);
return 400;
}

if (!GetFileSizeEx(srcfile, &dim)) {
sprintf(lastError, "Errore nel calcolo dimensione del file %s.\n", src);
CloseHandle(srcfile);
return 500;
}
CloseHandle(srcfile);

int ret = fileXOR(src, dst, (long long)dim.QuadPart, seed);

if (ret == 200 && !DeleteFile(src)) {
sprintf(lastError, "Errore nella cancellazione del file %s.\n", src);
return 500;
}

return ret;
}

int listFolder(char folder[], SOCKET sock) {
WIN32_FIND_DATA find_data;
char dir[MAX_PATH];
LARGE_INTEGER dim;
HANDLE handle_find = INVALID_HANDLE_VALUE;

snprintf(dir,MAX_PATH,"%s\\*.*",folder);
handle_find = FindFirstFile(dir,&find_data);

if(handle_find == INVALID_HANDLE_VALUE){
sprintf(lastError, "Errore apertura directory %s.\n", folder);
return 400;
}

do {
char path[PATHLEN];
char entry[PATHLEN+50];
ZeroMemory(path,sizeof(path));
ZeroMemory(entry, sizeof(entry));

if (strcmp(find_data.cFileName,".")==0 || strcmp(find_data.cFileName,"..")==0 || find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
continue;
}
else {
dim.LowPart = find_data.nFileSizeLow;
dim.HighPart = find_data.nFileSizeHigh;
snprintf(path,PATHLEN,"%s/%s",folder,find_data.cFileName);
snprintf(entry,PATHLEN+50,"%llu %s", dim.QuadPart, path);

sendMessage(sock, entry);
sendMessage(sock, "\r\n");
}
} while (FindNextFile(handle_find,&find_data)!=0);

FindClose(handle_find);
return 200;
}

int listRecursive(char folder[], SOCKET sock) {
WIN32_FIND_DATA find_data;
char dir[MAX_PATH];
LARGE_INTEGER dim;
HANDLE handle_find = INVALID_HANDLE_VALUE;

snprintf(dir, MAX_PATH, "%s\\*.*", folder);
handle_find = FindFirstFile(dir, &find_data);

if (handle_find == INVALID_HANDLE_VALUE) {
sprintf(lastError, "Errore apertura directory %s.\n", folder);
return 400;
}

do {
char path[PATHLEN];
char entry[PATHLEN+50];
ZeroMemory(path, sizeof(path));
ZeroMemory(entry, sizeof(entry));

if (strcmp(find_data.cFileName, ".") == 0 || strcmp(find_data.cFileName, "..") == 0) {
continue;
}
else if(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY){
sprintf(path, "%s/%s", folder, find_data.cFileName);
int ret = listRecursive(path,sock);
if (ret != 200)
return ret;
}
else {
dim.LowPart = find_data.nFileSizeLow;
dim.HighPart = find_data.nFileSizeHigh;
sprintf(path, "%s/%s", folder, find_data.cFileName);
sprintf(entry, "%llu %s", dim.QuadPart, path);

sendMessage(sock, entry);
sendMessage(sock, "\r\n");
}
} while (FindNextFile(handle_find, &find_data) != 0);

FindClose(handle_find);
return 200;
}

int parseRequest(char folder[], char message[], SOCKET sock) {
WIN32_FIND_DATA dirdata;
char temp[MAX_PATH];
snprintf(temp,MAX_PATH,"%s\\*.*",folder);
HANDLE dir = FindFirstFile(temp,&dirdata);
if (dir == INVALID_HANDLE_VALUE) {
return 1;
}
FindClose(dir);

int ret = 0;
if (strstr(message, "LSTF") != NULL) {
sendMessage(sock, STATE_PENDING);
ret = listFolder(folder, sock);
sendMessage(sock, "\r\n.\r\n");
}
else if (strstr(message, "LSTR") != NULL) {
sendMessage(sock, STATE_PENDING);
ret = listRecursive(folder, sock);
sendMessage(sock, "\r\n.\r\n");
}
else if (strstr(message, "ENCR") != NULL) {
char s[4] = "";
unsigned int seed = -1;
char path[PATHLEN] = "errore";

sscanf(message, "%s %u %[^\n]%*s", s, &seed, path);
if (seed != -1 && strcmp(path, "errore") != 0) {
ret = encrypt(path, seed, sock);
}
}
else if (strstr(message, "DECR") != NULL) {
char s[4] = "";
unsigned int seed = -1;
char path[PATHLEN] = "errore";

sscanf(message, "%s %u %[^\n]%*s", s, &seed, path);
if (seed != -1 && strcmp(path, "errore") != 0) {
ret = decrypt(path, seed, sock);
}
}

if (ret == 200) {
sendMessage(sock, STATE_OK);
}
else if (ret == 400) {
sendMessage(sock, lastError);
sendMessage(sock, STATE_ERROR);
}
else if (ret == 500) {
sendMessage(sock, lastError);
sendMessage(sock, STATE_UNAVAIL);
}

return ret;
}

int addRequest(SRWLOCK *mutex, CONDITION_VARIABLE *cond, char *folder, char *address, char *message, SOCKET sock) {
struct request *req = (struct request *)malloc(sizeof(struct request));
if (!req) {
char toLog[BUFSIZE] = "";
sprintf(toLog, "Errore malloc richiesta.\n");
writeLog(LOGFILE, toLog);
return 1;
}
AcquireSRWLockExclusive(mutex);

req->ID = nextReqID;
req->folder = folder;
req->address = address;
char buf[PATHLEN+100];
ZeroMemory(buf,sizeof(buf));
sprintf(buf,"%s",message);
req->message = buf;
req->sock = sock;
req->next = NULL;

char toLog[BUFSIZE] = "";
sprintf(toLog, "[Richiesta #%i] [%s] [%s]\n", nextReqID, address, message);
writeLog(LOGFILE, toLog);


if (numReqs == 0)
first = req;
else
last->next = req;
last = req;
numReqs++;

WakeAllConditionVariable(cond);
ReleaseSRWLockExclusive(mutex);

nextReqID++;
return 0;
}

struct request* removeRequest(SRWLOCK *mutex) {
struct request *req;

AcquireSRWLockExclusive(mutex);
if (numReqs>0) {
req = first;
first = req->next;
if (first == NULL)
last = NULL;
numReqs--;
}
else {
req = NULL;
}
ReleaseSRWLockExclusive(mutex);

return req;
}

DWORD WINAPI task(void *arg) {
int *threadID = (int *)arg;
struct request *req;

while (run) {
AcquireSRWLockExclusive(&reqMutex);
int r = numReqs;
ReleaseSRWLockExclusive(&reqMutex);

if (r>0) {
req = removeRequest(&reqMutex);
if (req!=NULL) {
char *folder = req->folder;
char *message = req->message;
SOCKET sock = req->sock;
int reqID = req->ID;

int ret = parseRequest(folder, message, sock);

char toLog[BUFSIZE] = "";
sprintf(toLog, "[Richiesta #%i] [Thread #%i: %i]\n", reqID, *threadID, ret);
writeLog(LOGFILE, toLog);

free(req);
closesocket(sock);
}
}
else {
AcquireSRWLockExclusive(&reqMutex);
SleepConditionVariableSRW(&reqCond,&reqMutex,INFINITE,0);
ReleaseSRWLockExclusive(&reqMutex);
}
}

return 0;
}

int executeServer(char folder[], unsigned short port, int threadNum) {
WIN32_FIND_DATA dirdata;
char temp[PATHLEN];
snprintf(temp,PATHLEN,"%s\\*.*",folder);
HANDLE dir = FindFirstFile(temp,&dirdata);
if (dir==INVALID_HANDLE_VALUE) {
char toLog[BUFSIZE] = "";
sprintf(toLog, "La cartella %s non e' una directory valida o non esiste.\n", folder);
writeLog(LOGFILE, toLog);
return 1;
}
FindClose(dir);

WSADATA wsaData;
int res;

SOCKET serverSock;

struct addrinfo *result = NULL;
struct addrinfo serveraddr;

int sendRes;
char message[BUFSIZE];
int msglen;
char strPort[6];
snprintf(strPort,6,"%hu",port);

res = WSAStartup(MAKEWORD(2,2), &wsaData);
if (res!=0) {
printf("Errore WSAStartup: %i\n",res);
return 1;
}

ZeroMemory(&serveraddr,sizeof(serveraddr));
serveraddr.ai_family = AF_INET;
serveraddr.ai_socktype = SOCK_STREAM;
serveraddr.ai_protocol = IPPROTO_TCP;
serveraddr.ai_flags = AI_PASSIVE;

res = getaddrinfo(NULL, strPort, &serveraddr, &result);
if (res!=0) {
printf("Errore getadddrinfo: %i\n",res);
WSACleanup();
return 1;
}

serverSock = socket(result->ai_family,result->ai_socktype,result->ai_protocol);
if (serverSock==INVALID_SOCKET) {
printf("Errore socket: %ld\n",WSAGetLastError());
freeaddrinfo(result);
WSACleanup();
return 1;
}

res = bind(serverSock,result->ai_addr,(int)result->ai_addrlen);
if (res==SOCKET_ERROR) {
printf("Errore bind: %d\n",WSAGetLastError());
freeaddrinfo(result);
closesocket(serverSock);
WSACleanup();
return 1;
}

freeaddrinfo(result);

res = listen(serverSock,SOMAXCONN);
if (res==SOCKET_ERROR) {
printf("Errore listen: %d\n",WSAGetLastError());
freeaddrinfo(result);
closesocket(serverSock);
WSACleanup();
return 1;
}

int *threadID = malloc(threadNum*sizeof(int));
HANDLE *threads = malloc(threadNum*sizeof(HANDLE));

for (int i = 0; i<threadNum; i++) {
threadID[i] = i;
threads[i] = CreateThread(NULL, 0, task, &threadID[i], 0, NULL);
}

SOCKET clientSock;
struct sockaddr_in clientAddr;
unsigned int clientlen = sizeof(clientAddr);

while (true) {
clientSock = accept(serverSock, (struct sockaddr *)&clientAddr, &clientlen);
if (clientSock == INVALID_SOCKET) {
printf("Errore accept: %d\n", WSAGetLastError());
freeaddrinfo(result);
closesocket(serverSock);
WSACleanup();
return 1;
}

char clientAddrReadable[NI_MAXHOST];
if (getnameinfo((const struct sockaddr *)&clientAddr, clientlen, clientAddrReadable, sizeof(clientAddrReadable), NULL, sizeof(NULL), NI_NUMERICHOST) != 0) {
printf("Errore risoluzione client.\n");
freeaddrinfo(result);
closesocket(serverSock);
WSACleanup();
return 1;
}

ZeroMemory(message, BUFSIZE);
msglen = recv(clientSock, message, BUFSIZE, 0);
if (addRequest(&reqMutex, &reqCond, folder, clientAddrReadable, message, clientSock) != 0) {
freeaddrinfo(result);
break;
}
}

closesocket(serverSock);
WSACleanup();

return 0;
}

void showHelp(char *command) {
printf("server~ ");
if (strcmp(command, "-h") != 0)
printf("Comando non valido.\n\t");
printf("Usage: {comando_1} [valore_1] ... {comando_n} [valore_n]\n\t\
Ogni valore e' marcato come opzionale, ma puo' essere obbligatorio a seconda del comando che lo precede.\n\n\t\
Comandi (valori obbligatori):\n\t\
-c\t obbligatorio, specifica la cartella di partenza\n\t\
\t ignora la voce folder=<dir/to/start/with>\n\t\
-p\t specifica la porta TCP sulla quale restare in ascolto; default: 8888\n\t\
\t ignora la voce port=<portNum>\n\t\
-n\t specifica il numero di thread da utilizzare; default: 1\n\t\
\t ignora la voce threadNumber=<threadNum>\n\n\t\
Comandi (nessun valore necessario):\n\t\
-h\t mostra questo messaggio\n\n\t\
Dettagli:\n\t\
Tutti i parametri possono essere definiti tramite il file misc/server.conf, ma ignorati se specificati tramite riga di comando.\n\t\
In particolare, l'opzione -c non e' obbligatoria se la cartella e' specificata in tale file.\n");

return;
}

int main(int argc, char *argv[]) {
BOOL r = CreateDirectory("misc",NULL);
if (r != TRUE && GetLastError() != ERROR_ALREADY_EXISTS) {
printf("Errore creazione directory di log.\n");
return 1;
}

FILE *srvlog = fopen(LOGFILE, "w");
if (srvlog == NULL) {
printf("Errore creazione file di log.\n");
return 1;
}
fclose(srvlog);

memset(folder,0,PATHLEN);
port = 0;
threadNum = -1;

loadConfig(&port, folder, &threadNum);

if (argc>1) {
for (int i = 1; i<argc; i++) {
if (strcmp(argv[i], "-c") == 0) {
if (i + 1<argc && strstr(argv[i + 1], "-") == NULL) {
memset(folder, 0, PATHLEN);
strncpy(folder, argv[i + 1], strlen(argv[i + 1]));
i++;
}
else {
showHelp(argv[i]);
}
}
else if (strcmp(argv[i], "-p") == 0) {
if (i + 1<argc && strstr(argv[i + 1], "-") == NULL) {
port = (unsigned short)atoi(argv[i + 1]);
i++;
}
else {
showHelp(argv[i]);
}
}
else if (strcmp(argv[i], "-n") == 0) {
if (i + 1<argc && strstr(argv[i + 1], "-") == NULL) {
threadNum = atoi(argv[i + 1]);
i++;
}
else {
showHelp(argv[i]);
}
}
else
showHelp(argv[i]);
}
}

if (strcmp(folder, "\0") == 0) {
showHelp(argv[0]);
return 1;
}

nextReqID = 0;
numReqs = 0;
InitializeSRWLock(&reqMutex);
InitializeConditionVariable(&reqCond);
run = true;

executeServer(folder,port,threadNum);

run = false;

return 0;
}
