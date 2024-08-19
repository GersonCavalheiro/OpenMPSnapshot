#include "Dissector.h"
#include "AhoCorasick.h"
#include "WuManber.h"
#include <string>
#include "../Util.h"
#include <omp.h>
using namespace std;
#include <sstream>

namespace patch
{
template < typename T > std::string to_string( const T& n )
{
std::ostringstream stm ;
stm << n ;
return stm.str() ;
}
}

#define _DISSECTOR_CHECK_OVERFLOW(a,b) \
do{ \
if(hdr!=NULL){ \
if(a>b) \
return; \
} \
}while(0)

struct length {
bool operator() ( const string& a, const string& b )
{
return a.size() < b.size();
}
};


int packetLength;

void Dissector::dissectEthernet(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){
uint8_t* onBoardProtocol;	

onBoardProtocol = (uint8_t *)packetPointer+ETHERNET_HEADER_SIZE; 

EthernetVirtualAction(packetPointer,totalHeaderLength,hdr,new Ethernet2Header(packetPointer),user);

*totalHeaderLength += ETHERNET_HEADER_SIZE;

DEBUG2("Ethernet packet");
switch(ntohs(((struct ether_header*)packetPointer)->type)){

case ETHER2_TYPE_IP4:
_DISSECTOR_CHECK_OVERFLOW(*totalHeaderLength+IP4_NO_OPTIONS_HEADER_SIZE,hdr->caplen);
dissectIp4(onBoardProtocol,totalHeaderLength,hdr,user);
break;
case ETHER2_TYPE_IP6:

default:
DEBUG2("NOT supported");
break;

}	

}





void Dissector::dissectIp4(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){
uint8_t* onBoardProtocol;	

onBoardProtocol =(uint8_t *)packetPointer+Ip4Header::calcHeaderLengthInBytes(packetPointer);

Ip4VirtualAction(packetPointer,totalHeaderLength,hdr,new Ip4Header(packetPointer),user);
Ip4VirtualActionnew(packetPointer,totalHeaderLength,hdr,new Ip4Header16(packetPointer),user);

*totalHeaderLength +=Ip4Header::calcHeaderLengthInBytes(packetPointer);

DEBUG2("IP4 packet");
switch(((struct ip4_header*)packetPointer)->protocol){

case IP_PROT_ICMP:
_DISSECTOR_CHECK_OVERFLOW(*totalHeaderLength+ICMP_HEADER_SIZE,hdr->caplen);

dissectIcmp(onBoardProtocol,totalHeaderLength,hdr,user);
break;
case IP_PROT_TCP:
_DISSECTOR_CHECK_OVERFLOW(*totalHeaderLength+TCP_NO_OPTIONS_HEADER_SIZE,hdr->caplen);
dissectTcp(onBoardProtocol,totalHeaderLength,hdr,user);
break;
case IP_PROT_UDP:
_DISSECTOR_CHECK_OVERFLOW(*totalHeaderLength+UDP_HEADER_SIZE,hdr->caplen);
dissectUdp(onBoardProtocol,totalHeaderLength,hdr,user);
break;
default:
DEBUG2("NOT supported");
break;

}
*totalHeaderLength = Ip4Header::totalPacketLength(packetPointer);
packetLength = ntohs(((struct ip4_header*)packetPointer)->totalLength) & 0x0000FFFF;
}




void Dissector::dissectTcp(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){

TcpVirtualAction(packetPointer,totalHeaderLength,hdr,new TcpHeader(packetPointer),user);

DEBUG2("After TCP Virtual Action");
/

}

long hashCal(const char* pattern, int  m, int offset) {
long h = 0;
for (int j = 0; j < m; j++)
{
h = (256 * h + pattern[offset + j]) % 997;
}
return h;
}

bool compare(string a,string b)
{
return a.size() < b.size();
}

void Dissector::searchWords(vector<string> arr, int k, string text)
{
int states = buildMatchingMachine(arr, k);
DEBUG2("Completed building machine, No Of States = %d",states);

int currentState = 0;

printf("Text Size = %d", text.size());
#pragma omp parallel for firstprivate(currentState)
for (int i = 0; i < text.size(); ++i)
{
printf("ThreadID = %d, currentState = %d",omp_get_thread_num(),currentState);
currentState = findNextState(currentState, text[i]);
for (int i = 0; i < text.size(); ++i)
{
currentState = findNextState(currentState, text[i]);

printf("out[currentState][0] = %d \n",out[currentState][0]);
if (out[currentState][0] == 0)
continue;


int outSize = out[currentState][0];

for (int j = 1; j <= outSize; ++j)
{
int patIndex = out[currentState][j];
if(patIndex>=k || patIndex<0) continue;
DEBUG2("In searchWords outIndex=%d currentState=%d patIndex=%d",j,currentState,out[currentState][j]);
long start = (long) i - arr[patIndex].size() + 1;
if(start >= text.size()) continue;
printf("Word %s appears from %d to %d",arr[patIndex].c_str(),start,i);
}
}
}
}

void Dissector::payLoadRabinKarp(char* packetPointer,vector<string> tmp) {
vector<int> mapHash(997,-1);
set<int> setlen;

int payLoadLength = packetLength - 40;

for(int i=0;i<tmp.size();i++)
setlen.insert(tmp[i].length());

for(int i=0;i<tmp.size();i++)
{
long patHash = hashCal(tmp[i].c_str(), tmp[i].size(),0);
mapHash[patHash] = i;
}

int q = 997;
int R = 256;

for(auto it= setlen.begin();it!=setlen.end();it++)
{
int m = *it;
int RM = 1;
for (int i = 1; i <= m-1; i++)
RM = (256 * RM) % 997;

if (m > payLoadLength) break;
int txtHash = hashCal((char*)packetPointer, m,0);

if ((mapHash[txtHash]>0) && memcmp((char*)packetPointer,
tmp[mapHash[txtHash]].c_str(),m)==0)
{ cout<<"Virus Pattern " << tmp[mapHash[txtHash]] <<" exists"<<endl; break;}

for (int j = m; j < payLoadLength; j++) {
txtHash = (txtHash + q - RM*packetPointer[j-m] % q) % q;
txtHash = (txtHash*R + packetPointer[j]) % q;

int offset = j - m + 1;
if ((mapHash[txtHash]>0) &&
memcmp((char*) (packetPointer + offset), tmp[mapHash[txtHash]].c_str(),m)==0)
{ cout<<"Virus Pattern " << tmp[mapHash[txtHash]] <<" exists"<<endl; break;}
}
}

}

void Dissector::dissectUdp(const uint8_t* packetPointer, unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){


UdpVirtualAction(packetPointer,totalHeaderLength,hdr, new UdpHeader(packetPointer),user);

*totalHeaderLength +=UDP_HEADER_SIZE;
DEBUG2("UDP packet");


}

void Dissector::dissectIcmp(const uint8_t* packetPointer, unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){

IcmpVirtualAction(packetPointer,totalHeaderLength,hdr,new IcmpHeader(packetPointer),user);

*totalHeaderLength +=ICMP_HEADER_SIZE;
DEBUG2("ICMP packet");

}





unsigned int Dissector::dissect(const uint8_t* packetPointer,const struct pcap_pkthdr* hdr,const int deviceDataLinkInfo,void* user,int noOfPatterns){
unsigned int totalHeaderLength = 0;
this->noOfPatterns = noOfPatterns;

DEBUG2("Starting dissection...");
switch(deviceDataLinkInfo){

case DLT_EN10MB: 
dissectEthernet(packetPointer,&totalHeaderLength,hdr,user);
break;

default: 
DEBUG2("Not implemented yet");
return -1;
break;
}
DEBUG2("End of dissection\n");
EndOfDissectionVirtualAction(&totalHeaderLength,hdr,user);


return totalHeaderLength;
}

