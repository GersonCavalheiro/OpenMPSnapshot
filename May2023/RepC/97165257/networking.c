#include "networking.h"
void WinSockStart() {
WSADATA wsa;
if (WSAStartup(MAKEWORD(2,2), &wsa) != 0)
{
LFATAL("WSAStartup", WSAGetLastError());
}
}
void WinSockClose() {
WSACleanup();
}
int TrySend(ZL_ulong IP, ZL_ushort port, ZL_cstring message, ZL_ulong max_ping) {
SOCKET s;
s = socket(AF_INET, SOCK_STREAM, 0);
if(s == INVALID_SOCKET)
{
ITOA_R(WSAGetLastError(), out);
LDEBUG("Error in TrySend socket: %s", out);
return 1;
}
ZL_ulong nonblocking = 1;
int ioctlsocket_result = ioctlsocket(s, FIONBIO, &nonblocking);
if (ioctlsocket_result != 0) {
LFATAL("TrySend - cannot change socket mode", 1);
}
struct sockaddr_in addr;
addr.sin_addr.s_addr = IP;
addr.sin_family = AF_INET;
addr.sin_port = htons(port);
LDEBUG("Trying %s...\n", inet_ntoa(addr.sin_addr));
int connect_result = connect(s, (struct sockaddr *)&addr, sizeof(addr));
if (connect_result < 0)
{
if (connect_result == SOCKET_ERROR) {
int error = WSAGetLastError();
if (error != WSAEWOULDBLOCK) {
ITOA_R(WSAGetLastError(), out);
LDEBUG("Error in TrySend connect: %s", out);
closesocket(s);
return 1;
}
} else {
LDEBUG("Error in TrySend connect: %d [NON-WSA]", connect_result);
closesocket(s);
return 1;
}
} else {
LFATAL("Error in TrySend - socket is blocking", 1);
}
fd_set fdset;
struct timeval tv;
FD_ZERO(&fdset);
FD_SET(s, &fdset);
tv.tv_sec = max_ping / 1000;
tv.tv_usec = (max_ping - (tv.tv_sec * 1000)) * 1000;
int select_result = select(0, NULL, &fdset, NULL, &tv);
if (select_result == 0) {
closesocket(s);
return 1;
} else if (select_result == SOCKET_ERROR) {
ITOA_R(WSAGetLastError(), out);
LDEBUG("Error in TrySend select: %s", out);
closesocket(s);
return 1;
} else {
if (send(s, message, strlen(message), 0) == SOCKET_ERROR)
{
ITOA_R(WSAGetLastError(), out);
LDEBUG("Error in TrySend send: %s", out);
closesocket(s);
return 1;
}
}
LDEBUG("Sent to %s!\n", inet_ntoa(addr.sin_addr));
closesocket(s);
return 0;
}
void TryBroadcastNetwork(ZL_ulong start, ZL_ulong len, ZL_ushort port, ZL_cstring message, ZL_ulong max_ping) {
LDEBUG("Starting scanning network %ldL, %ld elements\n", start, len);
#if DEBUG
clock_t currentTime = clock();
#endif
omp_set_dynamic(0);
omp_set_num_threads(len);
#pragma omp parallel default(none) shared(start, port, message, max_ping, len)
{
#pragma omp single
{
LDEBUG("Running %d threads\n", omp_get_num_threads());
}
#pragma omp for
for (ZL_ulong cur = 0; cur < len; ++cur) {
TrySend(start + SWAP_UINT32(cur), port, message, max_ping);
}
}
LDEBUG("TryBroadcastNetwork took %ld to complete\n", clock() - currentTime);
}
void TryBroadcastAllNetworks(ZL_ushort port, ZL_cstring message, ZL_ulong max_ping) {
PIP_ADAPTER_INFO pAdapterInfo;
PIP_ADAPTER_INFO pAdapter;
IP_ADDR_STRING* pAddrStruct;
ZL_ulong ulOutBufLen = 0;
GetAdaptersInfo(NULL, &ulOutBufLen);
pAdapterInfo = (IP_ADAPTER_INFO*) malloc(ulOutBufLen);
if (GetAdaptersInfo(pAdapterInfo, &ulOutBufLen) == NO_ERROR) {
pAdapter = pAdapterInfo;
while (pAdapter) {
pAddrStruct = &(pAdapter->IpAddressList);
while (pAddrStruct) {
ZL_ulong binaryIP = inet_addr(pAddrStruct->IpAddress.String);
ZL_ulong netmask = inet_addr(pAddrStruct->IpMask.String); 
if (   binaryIP != ULONG_MAX
&& binaryIP != 0
&& netmask != ULONG_MAX 
&& netmask != 0) {
ZL_ulong subnet = binaryIP & netmask;
ZL_ulong subnet_length = SWAP_UINT32(ULONG_MAX & (~netmask));
TryBroadcastNetwork(
subnet, subnet_length, 
port, message, max_ping
);
}
pAddrStruct = pAddrStruct->Next;
}
pAdapter = pAdapter->Next;
}
} else {
LDEBUG("Error: TryBroadcastAllNetworks failed completely");
}
SFREE(pAdapterInfo);
}