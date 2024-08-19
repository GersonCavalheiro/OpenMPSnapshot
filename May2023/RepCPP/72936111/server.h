#ifndef OMPSERVER_H
#define OMPSERVER_H

#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>          
#include <netdb.h>               
#include <unistd.h>              
#include <cstring>
#include <arpa/inet.h>           
#include <fcntl.h>               
#include <sys/time.h>            

#ifdef _OPENMP
#include <omp.h>                 
#endif

#include <vector>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <cstdio>
#include <map>


class OmpServer
{
private:

const static int MaximumClients     = 50;   
const static int NbQueuedClients    = 4;    
const static int BufferSize         = 256;  


sockaddr_in         address;    
int                 sock;       
std::vector< std::pair<int,std::string> >    clients;    
int                 nbClients;  
bool                hasToStop;  

enum commands{msg=1,all,add,join,cmsg};

struct directive                   
{
commands cmd;                        
std::string buffer;             
std::string dest,nick;
};

struct userinfo
{
std::string info,pswd="";

};

std::vector< std::vector< std::string > > channels;
int channels_count=0;
std::map<std::string, int> channel_names;




bool bind(const int inPort){
address.sin_family       = AF_INET;
address.sin_addr.s_addr  = INADDR_ANY;
address.sin_port         = htons ( inPort );

return IsValidReturn( ::bind( sock, (struct sockaddr*) &address, sizeof(address)) );
}


bool listen(){
return IsValidReturn( ::listen( sock, NbQueuedClients ) );
}


int accept(){
int sizeOfAddress = sizeof(address);
const int newClientSock = ::accept ( sock, (sockaddr*)&address, (socklen_t*)&sizeOfAddress );
return (newClientSock < 0 ? -1 : newClientSock);
}


void processClient(const int inClientSocket){

if( !addClient(inClientSocket) ){
IsValidReturn( write( inClientSocket, "Server is full!\n", 16) );
::close(inClientSocket);
return;
}

if( !IsValidReturn( write( inClientSocket, "Welcome : enter your nick :\n", 28)) ){
removeClient(inClientSocket);
return;
}

const size_t BufferSize = 256;
char recvBuffer[BufferSize];

const int flag = fcntl(inClientSocket, F_GETFL, 0);
fcntl(inClientSocket, F_SETFL, flag | O_NONBLOCK);

bool FirstTime=true;   


while( true ){
fd_set  readSet;        
FD_ZERO(&readSet);

FD_SET(inClientSocket, &readSet);

timeval timeout = {1, 0};  
select( inClientSocket + 1, &readSet, NULL, NULL, &timeout);

if( hasToStop ){
break;
}

memset(&recvBuffer, 0, sizeof(recvBuffer));

if( FD_ISSET(inClientSocket, &readSet) ){
FD_CLR(inClientSocket, &readSet);
const int lenghtRead = read(inClientSocket, recvBuffer, BufferSize);
if(lenghtRead <= 0){
removeClient(inClientSocket);
break;
}
else if(0 < lenghtRead){
if(FirstTime==true)
{
int userid=SearchClient(inClientSocket);

recvBuffer[strlen(recvBuffer)-1]='\0';

if(userid>=0){
clients[userid].second=recvBuffer;
}



if( !sendsaved(userid) )
std::cerr<<"error sending saved messages";

FirstTime=false;

show(userid,recvBuffer);           
show();
}
else
{
directive temp;
std::string temp_str=recvBuffer;
temp.nick=clients[SearchClient(inClientSocket)].second;
if(parse(temp_str,temp))
{
if(temp.cmd==add){
addChannel(temp.dest,temp.nick);
}
else if(temp.cmd==join){
addtochannel(temp.dest,temp.nick);
}
else
broadcast(temp);

show_chs();
show(temp);
}
}

}
}
}
}



void removeClient(const int inClientSocket){
#pragma omp critical(CriticalClient)
{
int socketPosition = -1;
for(int idxClient = 0 ; idxClient < nbClients ; ++idxClient){
if( clients[idxClient].first == inClientSocket ){
socketPosition = idxClient;
break;
}
}
if( socketPosition != -1 ){
::close(inClientSocket);
--nbClients;
clients[socketPosition] = clients[nbClients];
}
}
}


bool addClient(const int inClientSocket){
bool cliendAdded = false;
#pragma omp critical(CriticalClient)
{
if(nbClients != MaximumClients){
clients[nbClients++].first = inClientSocket;
cliendAdded = true;
}
}
return cliendAdded;
}

void broadcast(const directive& data){
if(data.cmd==all)
{
#pragma omp critical(CriticalClient)
{
for(int idxClient = 0 ; idxClient < nbClients ; ++idxClient){

std::string tempBuffer="msg from " +data.nick+ ": "+data.buffer;
if( write(clients[idxClient].first, tempBuffer.c_str(), std::strlen(tempBuffer.c_str()) ) <= 0 ){
::close(clients[idxClient].first);
--nbClients;
clients[idxClient] = clients[nbClients];
--idxClient;
}
}
}
}
else if(data.cmd==msg)
{
#pragma omp critical(CriticalClient)
{
bool found=false;
int idxClient=SearchClient(data.dest);
if(idxClient>=0)
{
std::string tempBuffer="msg from " +data.nick+ ": " +data.buffer;
if( write(clients[idxClient].first, tempBuffer.c_str(), std::strlen(tempBuffer.c_str()) ) <= 0 ){
::close(clients[idxClient].first);
--nbClients;
clients[idxClient] = clients[nbClients];
}
found =true;
}
if(found==false)
save(data.dest,"msg from " +data.nick+ ": " +data.buffer);
}
}
else if(data.cmd==cmsg)
{

if(channel_names.find(data.dest) !=channel_names.end())
{
int index=channel_names[data.dest];
for(int i=0;i<channels[index].size();i++)
{
bool found=false;
int idxClient=SearchClient(channels[index][i]);
std::cout<<"\n idx :"<<idxClient<<" \n";
#pragma omp critical(CriticalClient)
{
if(idxClient>=0)
{
std::string tempBuffer="msg from " +data.nick+ ": " +data.buffer;
if( write(clients[idxClient].first, tempBuffer.c_str(), std::strlen(tempBuffer.c_str()) ) <= 0 ){
::close(clients[idxClient].first);
--nbClients;
clients[idxClient] = clients[nbClients];
}
found =true;
}
}
if(found==false)
save(channels[index][i],"msg from " +data.nick+ ": " +data.buffer);
}
}
else
std::cerr<<"[ERROR] no such channel \n";
}

}


OmpServer(const OmpServer&){}
OmpServer& operator=(const OmpServer&){ return *this; }

int SearchClient(const int SockId)
{
for(int i=0;i<nbClients;i++)
if(clients[i].first==SockId)
return i;
return -1;
}
int SearchClient(const std::string name)
{
for(int i=0;i<nbClients;i++)
if(clients[i].second.compare(name)==0)
return i;
return -1;
}

bool addChannel(std::string name,std::string creator_nick)
{
std::cout << name.size() << "\n";
channel_names[name] = channels_count;
channels_count++;

channels.push_back( std::vector<std::string>(1,creator_nick) );


return true;
}
bool addtochannel(std::string channel_name,std::string newname)
{
std::map<std::string,int>::iterator tmp_id =channel_names.find(channel_name);

if(tmp_id !=channel_names.end())
channels[ tmp_id->second ].push_back(newname);
return true;
}

bool sendtochannel();



bool parse(std::string& buffer,directive &temp)
{
if(buffer.compare(0,4,"/msg")==0)
{
temp.cmd=msg;
buffer=buffer.substr(5);

temp.dest=buffer.substr(0,buffer.find(" "));
buffer=buffer.substr(buffer.find(" ")+1);

temp.buffer=buffer;

return true;
}
else if(buffer.compare(0,4,"/all")==0)
{
temp.cmd=all;
temp.buffer=buffer.substr(5);
return true;
}
else if(buffer.compare(0,4,"/add")==0)
{
temp.cmd=add;
buffer=buffer.substr(5);
temp.dest=buffer.substr(0,buffer.find("\n"));
return true;
}
else if(buffer.compare(0,5,"/join")==0)
{
temp.cmd=join;

buffer=buffer.substr(6);
temp.dest=buffer.substr(0,buffer.find("\n"));
return true;
}
else if(buffer.compare(0,5,"/cmsg")==0)
{
temp.cmd=cmsg;
buffer=buffer.substr(6);

temp.dest=buffer.substr(0,buffer.find(" "));
buffer=buffer.substr(buffer.find(" ")+1);

temp.buffer=buffer;
return true;
}
else
return false;
}

void save(const std::string name,const std::string msg)
{
std::string newname=name+".dat";
#pragma omp critical(FileUpdate)
{
std::ofstream fout( newname.c_str() , std::ios::out|std::ios::app);
fout<<msg.c_str();
fout.close();
}
}
bool sendsaved(const int userid)
{
bool failed=false;
std::string newname=clients[userid].second+".dat";
const size_t BufferSize = 256;
#pragma omp critical(FileUpdate)
{
if(check(newname))
{
std::ifstream fin(newname.c_str());
char buffer[BufferSize];
fin.read( buffer, BufferSize);
int lenghtRead= strlen( buffer );

do
{
std::cerr<<"file buffer :|"<<buffer<<"|\n";

if( write(clients[userid].first, buffer, lenghtRead) <= 0 ){
::close(clients[userid].first);
--nbClients;
clients[userid] = clients[nbClients];

failed=true;
}

fin.read( buffer, BufferSize);
int lenghtRead= strlen( buffer );
}while(!fin.eof());

if(std::remove(newname.c_str()) != 0){
std::cerr<<"error deleting file :"<<newname<<std::endl;
failed=true;
}
}

}

return !failed ;
}

int adduser(int userid)
{
return 0;
std::string pass;
if( write(clients[userid].first,"Enter your password :" , std::strlen( "Enter your password :" ) ) <= 0 ){
::close(clients[userid].first);
--nbClients;
clients[userid] = clients[nbClients];
}

}


inline bool check(const std::string name) {
struct stat buffer;   
return ( stat(name.c_str(), &buffer) == 0 ); 
}

void show(const directive t)
{
std::cout<<"cmd :"<<t.cmd<<"\n";
std::cout<<"dest :|"<<t.dest<<"|\n";
std::cout<<"buffer :"<<t.buffer<<"\n";
}
void show(const int userid,const char recvBuffer[])
{
std::cout<<"clients nick :"<<clients[userid].second<<"\n";
std::cout<<"recvBuffer :|"<<recvBuffer<<"|\n";
}
void show()
{
for(int i = 0 ; i < nbClients ; ++i)
{
std::cout<<"client sockid :"<<clients[i].first<<"\n";
std::cout<<"client nickname:|"<<clients[i].second<<"|\n";
}
}
void show_chs()
{
std::map<std::string,int>::const_iterator it=channel_names.begin();
while(it!=channel_names.end())
{
std::cout<<"ch name |"<<it->first<<"| id :"<<it->second<<"\n";
it++;
}
std::cout<<"size :"<<channels.size()<<"\n";
for (unsigned int i = 0; i < channels.size(); ++i){
std::cout<<"size[i] :"<<channels[i].size()<<"\n";
std::cout << "ch no :"<<i<<" :";
for (unsigned int j = 0; j < channels[i].size(); ++j){
std::cout <<"i,j="<<i<<","<<j<<" |"<<channels[i][j]<<"| ";
std::cout<<"\n";
}
std::cout << std::endl;
}
}


static bool IsValidReturn(const int inReturnedValue){
return inReturnedValue >= 0;
}

public:



OmpServer(const int inPort)
: sock(-1), clients(MaximumClients,std::make_pair(-1,"")), nbClients(0), hasToStop(true) {
memset(&address, 0, sizeof(address));

if( IsValidReturn( sock = socket(AF_INET, SOCK_STREAM, 0)) ){
const int optval = 1;
if( !IsValidReturn(setsockopt( sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&optval, sizeof(optval)) ) ){
close();
}

if( !bind(inPort) ){
close();
}

if( !listen() ){
close();
}

const int flag = fcntl(sock, F_GETFL, 0);
fcntl(sock, F_SETFL, flag | O_NONBLOCK);
}
}


virtual ~OmpServer(){
close();
}

bool isValid() const {
return sock != -1;
}


void stopRun(){
hasToStop = true;
}


bool close(){
if(isValid()){           
const bool noErrorCheck = IsValidReturn(::close(sock));
sock = -1;
return noErrorCheck;
}
return false;
}



bool run(){
if( omp_in_parallel() || !isValid() ){
return false;
}

hasToStop = false;
#pragma omp parallel num_threads(MaximumClients)
{   
#pragma omp single nowait
{
while( !hasToStop ){
const int newSocket = accept();
if( newSocket != -1 ){
#pragma omp task untied
{
processClient(newSocket);
}
}
else{
usleep(2000);       
}
}
#pragma omp taskwait
}
}

#pragma omp parallel for
for(int idxClient = 0 ; idxClient < nbClients ; ++idxClient ){
::close(clients[idxClient].first);
}
nbClients = 0;

return true;
}
};

#endif 