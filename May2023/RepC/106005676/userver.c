#include "userver.h"
int XOR(int a, int b){
return a^b;
}
int fileXOR(char srcfile[], char dstfile[], long dim, int seed){
int src=open(srcfile,O_RDWR);
if(src<0){
sprintf(lastError,"Errore apertura file %s.\n",srcfile);
return 400;
}
int dst=open(dstfile,O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
if(dst<0){
sprintf(lastError,"Errore apertura file %s.\n",dstfile);
close(src);
return 400;
}
if(lockf(src,F_TLOCK,0)<0){
sprintf(lastError,"Errore lock su file %s.\n",srcfile);
close(src);
close(dst);
return 500;
}
if(lockf(dst,F_TLOCK,0)<0){
sprintf(lastError,"Errore lock su file %s.\n",dstfile);
lockf(src,F_ULOCK,0);
close(src);
close(dst);
return 500;
}
int result=lseek(dst, dim-1, SEEK_SET);
if (result==-1) {
sprintf(lastError,"Errore stretch file %s.\n",dstfile);
lockf(src,F_ULOCK,0);
lockf(dst,F_ULOCK,0);
close(src);
close(dst);
return 500;
}
result=write(dst,"",1);
if (result!=1) {
sprintf(lastError,"Errore scrittura su file %s.\n",dstfile);
lockf(src,F_ULOCK,0);
lockf(dst,F_ULOCK,0);
close(src);
close(dst);
return 500;
}
srand(seed);
if(dim<=256*1024){
long freePages=sysconf(_SC_AVPHYS_PAGES);
long pageDim=sysconf(_SC_PAGESIZE);
long freeMem=freePages*pageDim;
if(freeMem<=3*dim){
sprintf(lastError,"RAM insufficiente per aprire il file %s.\n",srcfile);
lockf(src,F_ULOCK,0);
lockf(dst,F_ULOCK,0);
close(src);
close(dst);
return 500;
}
char *srcmap=(char *)mmap(NULL,dim,PROT_READ,MAP_PRIVATE,src,0);
if((void *)srcmap==MAP_FAILED){
sprintf(lastError,"Errore file mapping su file %s.\n",srcfile);
lockf(src,F_ULOCK,0);
lockf(dst,F_ULOCK,0);
close(src);
close(dst);
return 500;
}
char *dstmap=(char *)mmap(NULL,dim,PROT_READ | PROT_WRITE,MAP_SHARED,dst,0);
if((void *)dstmap==MAP_FAILED){
sprintf(lastError,"Errore file mapping su file %s.\n",dstfile);
munmap((void *)srcmap,dim);
lockf(src,F_ULOCK,0);
lockf(dst,F_ULOCK,0);
close(src);
close(dst);
return 500;
}
long keyDim=(long)ceil((double)dim/4)*4;
int key[keyDim];
for(long i=0;i<keyDim;i++){
key[i]=rand()%65536; 
}
long i,j;
for(i=0,j=0;i<dim && j<keyDim;i+=4,j++){
dstmap[i]=(char)(XOR((int)srcmap[i],key[j]));
dstmap[i+1]=(char)(XOR((int)srcmap[i+1],key[j]));
dstmap[i+2]=(char)(XOR((int)srcmap[i+2],key[j]));
dstmap[i+3]=(char)(XOR((int)srcmap[i+3],key[j]));
}
munmap((void *)srcmap,dim);
munmap((void *)dstmap,dim);
}
else{
long fiveMB=5*pow(2,20);
int chunks=(int)ceil((double)dim/fiveMB);
for(int c=0;c<chunks;c++){
long freePages=sysconf(_SC_AVPHYS_PAGES);
long pageDim=sysconf(_SC_PAGESIZE);
long freeMem=freePages*pageDim;
if(freeMem<=2*fiveMB){
sprintf(lastError,"RAM insufficiente per aprire il file %s.\n",srcfile);
lockf(src,F_ULOCK,0);
lockf(dst,F_ULOCK,0);
close(src);
close(dst);
return 500;
}
long start=(c)*fiveMB;
long end=(c+1)*fiveMB;
long realEnd=end;
if(dim<realEnd)
realEnd=dim;
long chunkDim=realEnd-start;
if(dim-start<chunkDim)
chunkDim=dim-start;
char *srcmap=(char *)mmap(NULL,chunkDim,PROT_READ,MAP_PRIVATE,src,start);
if((void *)srcmap==MAP_FAILED){
sprintf(lastError,"Errore file mapping su file %s, chunk #%i.\n",srcfile,c);
lockf(src,F_ULOCK,0);
lockf(dst,F_ULOCK,0);
close(src);
close(dst);
return 500;
}
char *dstmap=(char *)mmap(NULL,chunkDim,PROT_READ | PROT_WRITE,MAP_SHARED,dst,start);
if((void *)dstmap==MAP_FAILED){
sprintf(lastError,"Errore file mapping su file %s, chunk #%i.\n",dstfile,c);
munmap((void *)srcmap,chunkDim);
lockf(src,F_ULOCK,0);
lockf(dst,F_ULOCK,0);
close(src);
close(dst);
return 500;
}
int mpThreads=(int)ceil((double)chunkDim/(256*1024));
long keyDimT = (long)ceil((double)chunkDim / (mpThreads * 4));
int key[mpThreads][keyDimT];
for(long j=0;j<mpThreads;j++){
for(long i=0;i<keyDimT;i++){
key[j][i]=rand()%65536; 
}
}
#pragma omp parallel num_threads(mpThreads)
{
int threadID=omp_get_thread_num();
int min=(threadID)*256*1024;
int max=(threadID+1)*256*1024;
for(long i=min;i<max && i<chunkDim;i+=4){
int val=key[threadID][(i-min)/4];
dstmap[i]=(char)(XOR((int)srcmap[i],val));
dstmap[i+1]=(char)(XOR((int)srcmap[i+1],val));
dstmap[i+2]=(char)(XOR((int)srcmap[i+2],val));
dstmap[i+3]=(char)(XOR((int)srcmap[i+3],val));
}
}
munmap((void *)srcmap,chunkDim);
munmap((void *)dstmap,chunkDim);
}
}
lockf(src,F_ULOCK,0);
lockf(dst,F_ULOCK,0);
close(src);
close(dst);
return 200;
}
int sendMessage(int sock, char message[]){
char buf[BUFSIZE];
memset(buf,0,BUFSIZE);
strncpy(buf,message,BUFSIZE);
int msglen=write(sock,buf,BUFSIZE);
if(msglen<0){
char toLog[BUFSIZE]="";
sprintf(toLog,"Errore write sul socket.\n");
writeLog(LOGFILE,toLog);
return 1;
}
return 0;
}
int encrypt(char src[], int seed, int sock){
char dst[PATHLEN]="";
strncpy(dst,src,strlen(src));
strncat(dst,"_enc",5);
long dim=-1;
struct stat st;
if(stat(src, &st) == 0)
dim=st.st_size;
if(dim==-1){
if(errno==ENOENT){
sprintf(lastError,"File %s non esistente.\n",src);
return 400;
}
else{
sprintf(lastError,"Errore nel calcolo dimensione del file %s.\n",src);
return 500;
}
}
int ret=fileXOR(src,dst,dim,seed);
if(ret==200 && unlink(src)){
sprintf(lastError,"Errore nella cancellazione del file %s.\n",src);
return 500;
}
return ret;
}
int decrypt(char src[], int seed, int sock){
char *enc=NULL;
char *temp = strstr(src, "_enc");
while (temp) {
enc = temp++;
temp = strstr(temp, "_enc");
}
if(enc==NULL || strlen(enc)!=4){
sprintf(lastError,"Il file %s non e' un file cifrato.\n",src);
return 400;
}
char dst[PATHLEN]="";
strncpy(dst,src,strlen(src)-4);
long dim=-1;
struct stat st;
if(stat(src, &st) == 0)
dim=st.st_size;
if(dim==-1){
if(errno==ENOENT){
sprintf(lastError,"File %s non esistente.\n",src);
return 400;
}
else{
sprintf(lastError,"Errore nel calcolo dimensione del file %s.\n",src);
return 500;
}
}
int ret=fileXOR(src,dst,dim,seed);
if(ret==200 && unlink(src)){
sprintf(lastError,"Errore nella cancellazione del file %s.\n",src);
return 500;
}
return ret;
}
int listFolder(char folder[], int sock){
DIR* dir=opendir(folder);
if(dir==NULL){
sprintf(lastError,"Errore apertura directory %s.\n",folder);
return 400;
}
while(true){
struct dirent *val=NULL;
char path[PATHLEN]="";
char entry[PATHLEN+50]="";
memset(path,0,sizeof(path));
memset(entry,0,sizeof(entry));
val=readdir(dir);
if(val==NULL){
break;
}
if(strcmp(val->d_name,".")==0 || strcmp(val->d_name,"..")==0 || (val->d_type & DT_DIR))
continue;
strncpy(path,folder,PATHLEN);
if(strstr(path+(strlen(path)-1),"/")==NULL)
strncat(path,"/",1);
strncat(path,val->d_name,PATHLEN-strlen(path));
long dim=-1;
struct stat st;
if(stat(path, &st) == 0)
dim=st.st_size;
if(dim==-1){
sprintf(lastError,"Errore nel calcolo dimensione del file %s.\n",path);
return 500;
}
sprintf(entry,"%li %s",dim,path);
sendMessage(sock,entry);
sendMessage(sock,"\r\n");
}
return 200;
}
int listRecursive(char folder[], int sock){
DIR* dir=opendir(folder);
if(dir==NULL){
sprintf(lastError,"Errore apertura directory %s.\n",folder);
return 400;
}
while(true){
struct dirent *val=NULL;
char path[PATHLEN]="";
char entry[PATHLEN+50]="";
memset(path,0,sizeof(path));
memset(entry,0,sizeof(entry));
val=readdir(dir);
if(val==NULL){
break;
}
if(strcmp(val->d_name,".")==0 || strcmp(val->d_name,"..")==0)
continue;
strncpy(path,folder,PATHLEN);
if(strstr(path+(strlen(path)-1),"/")==NULL)
strncat(path,"/",1);
strncat(path,val->d_name,PATHLEN-strlen(path));
if(!(val->d_type & DT_DIR)){
long dim=-1;
struct stat st;
if(stat(path, &st) == 0)
dim=st.st_size;
if(dim==-1){
sprintf(lastError,"Errore nel calcolo dimensione del file %s.\n",path);
return 500;
}
sprintf(entry,"%li %s",dim,path);
}
else{
if(dir){
int ret=listRecursive(path,sock);
if(ret!=200)
return ret;
}
else{
sprintf(lastError,"Errore nell'apertura della directory %s.\n",path);
return 500;
}
}
if(strcmp(entry,"")!=0)
sendMessage(sock,entry);
sendMessage(sock,"\r\n");
}
closedir(dir);
return 200;
}
int parseRequest(char folder[], char message[], int sock){
DIR* dir=opendir(folder);
if(dir==NULL){
return 1;
}
int ret=0;
if(strstr(message,"LSTF")!=NULL){
sendMessage(sock,STATE_PENDING);
ret=listFolder(folder,sock);
sendMessage(sock,"\r\n.\r\n");
}
else if(strstr(message,"LSTR")!=NULL){
sendMessage(sock,STATE_PENDING);
ret=listRecursive(folder,sock);
sendMessage(sock,"\r\n.\r\n");
}
else if(strstr(message,"ENCR")!=NULL){
char s[4]="";
unsigned int seed=-1;
char path[PATHLEN]="errore";
sscanf(message,"%s %u %[^\n]%*s",s,&seed,path);
if(seed!=-1 && strcmp(path,"errore")!=0){
ret=encrypt(path,seed,sock);
}
}
else if(strstr(message,"DECR")!=NULL){
char s[4]="";
unsigned int seed=-1;
char path[PATHLEN]="errore";
sscanf(message,"%s %u %[^\n]%*s",s,&seed,path);
if(seed!=-1 && strcmp(path,"errore")!=0){
ret=decrypt(path,seed,sock);
}
}
if(ret==200){
sendMessage(sock,STATE_OK);
}
else if(ret==400){
sendMessage(sock,lastError);
sendMessage(sock,STATE_ERROR);
}
else if(ret==500){
sendMessage(sock,lastError);
sendMessage(sock,STATE_UNAVAIL);
}
return ret;
}
int addRequest(pthread_mutex_t *mutex,pthread_cond_t *cond,char *folder,char *address,char *message,int sock){
struct request *req=(struct request *)malloc(sizeof(struct request));
if(!req){
char toLog[BUFSIZE]="";
sprintf(toLog,"Errore malloc richiesta.\n");
writeLog(LOGFILE,toLog);
return 1;
}
pthread_mutex_lock(mutex);
req->ID=nextReqID;
req->folder=folder;
req->address=address;
req->message=message;
req->sock=sock;
req->next=NULL;
char toLog[BUFSIZE]="";
sprintf(toLog,"[Richiesta #%i] [%s] [%s]\n",nextReqID,address,message);
writeLog(LOGFILE,toLog);
if(numReqs==0)
first=req;
else
last->next=req;
last=req;
numReqs++;
pthread_cond_broadcast(cond);
pthread_mutex_unlock(mutex);
nextReqID++;
return 0;
}
struct request* removeRequest(pthread_mutex_t *mutex){
struct request *req;
pthread_mutex_lock(mutex);
if(numReqs>0){
req=first;
first=req->next;
if(first==NULL)
last=NULL;
numReqs--;
}
else{
req=NULL;
}
pthread_mutex_unlock(mutex);
return req;
}
void *task(void *arg){
int *threadID=(int *)arg;
struct request *req;
while(run){
pthread_mutex_lock(&reqMutex);
int r=numReqs;
pthread_mutex_unlock(&reqMutex);
if(r>0){
pthread_mutex_lock(&reqMutex);
req=removeRequest(&reqMutex);
pthread_mutex_unlock(&reqMutex);
if(req){
char *folder=req->folder;
char *message=req->message;
int sock=req->sock;
int reqID=req->ID;
int ret=parseRequest(folder, message, sock);
char toLog[BUFSIZE]="";
sprintf(toLog,"[Richiesta #%i] [Thread #%i: %i]\n",reqID,*threadID,ret);
writeLog(LOGFILE,toLog);
free(req);
close(sock);
}
}
else{
pthread_mutex_lock(&reqMutex);
pthread_cond_wait(&reqCond,&reqMutex);
pthread_mutex_unlock(&reqMutex);
}
}
return NULL;
}
int executeServer(char folder[], unsigned short port, int threadNum){
DIR* dir=opendir(folder);
if(dir){
closedir(dir);
int serverSock;
struct sockaddr_in serveraddr;
int optval;
int msglen;
serverSock=socket(AF_INET, SOCK_STREAM, 0);
if(serverSock<0){
char toLog[BUFSIZE]="";
sprintf(toLog,"Errore apertura socket.\n");
writeLog(LOGFILE,toLog);
return 1;
}
optval=1;
setsockopt(serverSock, SOL_SOCKET, SO_REUSEADDR, (const void*)&optval, sizeof(int));
memset((char *)&serveraddr,0,sizeof(serveraddr));
serveraddr.sin_family = AF_INET;
serveraddr.sin_addr.s_addr=htonl(INADDR_ANY);
serveraddr.sin_port=htons(port);
if(bind(serverSock, (struct sockaddr *)&serveraddr, sizeof(serveraddr))<0){
char toLog[BUFSIZE]="";
sprintf(toLog,"Errore binding sul socket.\n");
writeLog(LOGFILE,toLog);
return 1;
}
if(listen(serverSock,5)<0){
char toLog[BUFSIZE]="";
sprintf(toLog,"Errore listening sul socket.\n");
writeLog(LOGFILE,toLog);
return 1;
}
int threadID[threadNum];
pthread_t threads[threadNum];
for(int i=0;i<threadNum;i++){
threadID[i]=i;
pthread_create(&threads[i],NULL,task,(void *)&threadID[i]);
}
int clientSock;
struct sockaddr_in clientAddr;
char message[BUFSIZE];
unsigned int clientlen=sizeof(clientAddr);
while(true){
clientSock=accept(serverSock, (struct sockaddr *)&clientAddr, &clientlen);
if(clientSock<0){
break;
}
char clientAddrReadable[NI_MAXHOST];
if (getnameinfo((const struct sockaddr *)&clientAddr, clientlen, clientAddrReadable, sizeof(clientAddrReadable), NULL, sizeof(NULL), NI_NUMERICHOST) != 0){
char toLog[BUFSIZE]="";
sprintf(toLog,"Errore risoluzione client.\n");
writeLog(LOGFILE,toLog);
break;
}
memset(message,0,BUFSIZE);
msglen=read(clientSock,message,BUFSIZE);
if(msglen<0){
char toLog[BUFSIZE]="";
sprintf(toLog,"Errore read sul socket.\n");
writeLog(LOGFILE,toLog);
break;
}
if(addRequest(&reqMutex,&reqCond,folder,clientAddrReadable,message,clientSock)!=0){
break;
}
}
close(serverSock);
for(int i=0;i<threadNum;i++){
pthread_join(threads[i],NULL);
}
}
else if(ENOENT == errno || ENOTDIR == errno){
char toLog[BUFSIZE]="";
sprintf(toLog,"La cartella %s non e' una directory valida o non esiste.\n",folder);
writeLog(LOGFILE,toLog);
return 1;
}
return 0;
}
void showHelp(char *command){
printf("server~ ");
if(strcmp(command,"-h")!=0)
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
int main(int argc, char *argv[]){
int r=mkdir("misc",0777);
if(r!=0 && errno!=EEXIST){
printf("Errore creazione directory di log.\n");
return 1;
}
FILE *srvlog=fopen(LOGFILE,"w");
if(srvlog==NULL){
printf("Errore creazione file di log.\n");
return 1;
}
fclose(srvlog);
memset(folder,0,PATHLEN);
port=0;
threadNum=-1;
loadConfig(&port,folder,&threadNum);
if(argc>1){
for(int i=1;i<argc;i++){
if(strcmp(argv[i],"-c")==0){
if(i+1<argc && strstr(argv[i+1],"-")==NULL){
memset(folder,0,PATHLEN);
strncpy(folder,argv[i+1],strlen(argv[i+1]));
i++;
}
else{
showHelp(argv[i]);
}
}
else if(strcmp(argv[i],"-p")==0){
if(i+1<argc && strstr(argv[i+1],"-")==NULL){
port=(unsigned short)atoi(argv[i+1]);
i++;
}
else{
showHelp(argv[i]);
}
}
else if(strcmp(argv[i],"-n")==0){
if(i+1<argc && strstr(argv[i+1],"-")==NULL){
threadNum=atoi(argv[i+1]);
i++;
}
else{
showHelp(argv[i]);
}
}
else
showHelp(argv[i]);
}
}
if(strcmp(folder,"\0")==0){
showHelp(argv[0]);
return 1;
}
makeDaemon();
struct sigaction sa;
memset((char *)&sa,0,sizeof(sa));
sa.sa_handler=sigHandler;
if(sigaction(SIGHUP, &sa, NULL)<0){
char toLog[BUFSIZE]="";
sprintf(toLog,"Errore sigaction\n");
writeLog(LOGFILE,toLog);
return 1;
}
nextReqID=0;
numReqs=0;
reqMutex=(pthread_mutex_t)PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
reqCond=(pthread_cond_t)PTHREAD_COND_INITIALIZER;
run=true;
while(true){
executeServer(folder,port,threadNum);
if(loadConfig(&port,folder,&threadNum)!=0){
char toLog[BUFSIZE]="";
sprintf(toLog,"Errore lettura del file di configurazione.\n");
writeLog(LOGFILE,toLog);
return 1;
}
run=true;
}
return 0;
}
static void makeDaemon(){
pid_t pid = fork();
if(pid<0){
printf("Errore fork\n");
exit(EXIT_FAILURE);
}
if(pid>0)
exit(EXIT_SUCCESS);
if(setsid()<0)
exit(EXIT_FAILURE);
struct sigaction sa;
memset((char *)&sa,0,sizeof(sa));
sa.sa_handler=sigIgnorer;
if(sigaction(SIGHUP, &sa, NULL)<0){
printf("Errore sigaction.\n");
exit(EXIT_FAILURE);
}
if(sigaction(SIGCHLD, &sa, NULL)<0){
printf("Errore sigaction.\n");
exit(EXIT_FAILURE);
}
pid=fork();
if(pid<0){
printf("Errore fork\n");
exit(EXIT_FAILURE);
}
if(pid>0)
exit(EXIT_SUCCESS);
pid=getpid();
char toLog[BUFSIZE]="";
sprintf(toLog,"Server avviato. PID: %ld\n",(long)pid);
writeLog(LOGFILE,toLog);
for(int i=sysconf(_SC_OPEN_MAX);i>=0;i--){
close(i);
}
}
void sigIgnorer(int signal){
}
void sigHandler(int signal){
run=false;
pthread_mutex_lock(&reqMutex);
pthread_cond_broadcast(&reqCond);
pthread_mutex_unlock(&reqMutex);
}
