#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <memory.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <omp.h>
u_int64_t _cellVal = 0;
u_int64_t _traceSteps = 0;
double _totalTime = 0;
double _totalCellTime = 0;
double _totalTraceTime = 0;
int _MAX_SIMILARITY;
int _counterMax;
int Q_len = 0, D_len = 0 ;
int MATCH;
int MISMATCH;
int GAP;
int THREADS;
bool nameBool, inputBool, matchBool, misBool, gapBool,threadBool = false;
int pairs= -1;
int qMin = -1;
int qMax = -1;
int dSize = -1;
char *INPUT;
char *REPORT;
char threadC[2];
char cwd[512];
int antiDiagNum = 1 ;
FILE *finFile;
int **ScoreTable = NULL;
char * Q = NULL;
char * D = NULL;
const char* inVariable[] = {"-name", "-input", "-match", "-mismatch", "-gap" , "-threads"};
void terminalPrinter(){
printf("\n\n");
printf("NUMBER OF Q-D PAIRS: %d\n", pairs);
printf("TOTAL NUMBER OF CELLS UPDATED: %lu\n",_cellVal);
printf("TRACEBACK STEPS: %lu\n",_traceSteps);
printf("TOTAL TIME : %lf\n", _totalTime);
printf("TOTAL CELL TIME : %lf\n", _totalCellTime);
printf("TOTAL TRACEBACK TIME : %lf\n", _totalTraceTime);
printf("CUPS TOTAL TIME: %.3lf\n", _cellVal/_totalTime);
printf("CUPS CELL TIME: %.3lf\n", _cellVal/_totalCellTime);
}
double gettime(void)
{
struct timeval ttime;
gettimeofday(&ttime, NULL);
return ttime.tv_sec+ttime.tv_usec * 0.000001;
}
void finFileDir(){
if(getcwd(cwd, sizeof(cwd))==NULL){
printf("Error while trying to get current directory");
exit(-1);
}
sprintf(threadC,"%d",THREADS);
strcat(cwd,"/Report_");
strcat(cwd,REPORT);
strcat(cwd,"_OMP_");
strcat(cwd,threadC);
strcat(cwd,".txt");
}
void commandChecker(int argc, char * argv[]){
for (uint8_t i = 0; i < argc; i++) {
if(strcmp(argv[i],inVariable[0]) == 0){
REPORT = argv[++i];
nameBool = true;
}
if(strcmp(argv[i],inVariable[1]) == 0){
INPUT = argv[++i];
if(strcmp(&INPUT[strlen(INPUT)-4],".txt")!=0){
printf("Your input path does not end in .txt . Exiting.\n");
exit(-9);
}
inputBool = true;
}
if(strcmp(argv[i],inVariable[2]) == 0) {
MATCH = atoi(argv[++i]);
matchBool = true;
}
if(strcmp(argv[i],inVariable[3]) == 0) {
MISMATCH = atoi(argv[++i]);
misBool = true;
}
if(strcmp(argv[i],inVariable[4]) == 0) {
GAP = atoi(argv[++i]);
gapBool = true;
}
if(strcmp(argv[i],inVariable[5]) == 0) {
THREADS = atoi(argv[++i]);
threadBool = true;
}
}
if( !nameBool | !inputBool | !matchBool | !misBool | !gapBool | !threadBool){
printf("NOT ALL DEMANDED INPUT VARIABLES WERE GIVEN IN COMMAND LINE. EXITING.......");
exit(-10);
}
}
void ErrorCode(int checker){
if(checker < 0){
printf("Error code . The testing file does not have the requested format. Terminating\n");
exit(-9);
}
}
char* reverseArr(char *str, size_t len) {
size_t i = 0;
while (len > i) {
char tmp = str[--len];
str[len] = str[i];
str[i++] = tmp;
}
return str;
}
void fileHeaderValues(FILE *fp){
fscanf(fp, "Pairs: %d\n", &pairs);
ErrorCode(pairs);
fscanf(fp, "Q_Sz_Min: %d\n", &qMin);
ErrorCode(qMin);
fscanf(fp, "Q_Sz_Max: %d\n", &qMax);
ErrorCode(qMax);
fscanf(fp, "D_Sz_All: %d\n", &dSize);
ErrorCode(dSize);
}
int antiDiagLength(int currAnti){
int minFinder = D_len < Q_len ? D_len : Q_len; 
int maxFinder = D_len < Q_len ? Q_len : D_len;
if(currAnti < D_len && currAnti < Q_len){
return currAnti;
}else if(currAnti > D_len && currAnti > Q_len ){
return minFinder  - (currAnti - maxFinder  ) ;
}else{
return  minFinder;
}
}
int updateScore(long long int x, long long int y){
int up, left, diag;
up = ScoreTable[x-1][y] + GAP;
left = ScoreTable[x][y-1] + GAP; 
int tempDiag,tempMax;
int tempCell=0;
tempDiag = (D[y - 1] == Q[x - 1]) ? MATCH : MISMATCH;
diag = ScoreTable[x - 1][y - 1] + tempDiag;
if( up <= 0 && left <= 0 && diag <= 0 ){    
tempMax = 0;
}else{
if( up > left && up > diag){
tempMax = up;
}else if (left >= up && left > diag){
tempMax = left;
}else{
tempMax = diag;
}
tempCell++;
}
ScoreTable[x][y] = tempMax;
if(tempMax > _MAX_SIMILARITY) {
_counterMax = 0;
_MAX_SIMILARITY = tempMax;
}else if(tempMax == _MAX_SIMILARITY && tempMax!=0){
_counterMax++;
}
return tempCell;
}
void dataParser(){
fprintf(finFile, "\n\nQ: \t%s", Q);
fprintf(finFile, "\nD: \t%s\n", D);
Q_len = strlen(Q);
D_len = strlen(D);
ScoreTable = malloc((Q_len+1)* sizeof(int *));
if(ScoreTable ==NULL){
printf("Could not allocate memory for matrix ScoreTable.Terminating");
exit(-2);
}
for(int i = 0; i< Q_len+1; i++){
ScoreTable[i] = malloc((D_len+1)* sizeof(int));
if(ScoreTable[i] == NULL){
printf("Could not allocate memory for matrix ScoreTable.Terminating");
exit(-2);
}
}
for(int i=0; i < Q_len + 1; i++){
ScoreTable[i][0] = 0;
}
for(int i=0; i < D_len + 1; i++){
ScoreTable[0][i] = 0;
}
_counterMax = 0;
_MAX_SIMILARITY=0;
antiDiagNum = D_len + Q_len  - 1;
double cellTimeInit = gettime();
int diaLen,initX, initY,sum;
for ( int i = 1; i<=antiDiagNum; i++){
sum = 0;
diaLen = antiDiagLength(i);
if(i <= Q_len){
initX = i;
initY = 1;
}else{
initX = Q_len;
initY = i - Q_len + 1;
}
int j=0;
#pragma omp parallel for num_threads(THREADS) default(none)shared(initX,initY,diaLen,THREADS) private(j) reduction(+:sum)
for(j =0;j< diaLen;j++){
sum +=updateScore(initX - j,initY + j);
}
_cellVal+=sum;
}
double cellTimeFin = gettime();
_totalCellTime+= (cellTimeFin-cellTimeInit);
long xMax[_counterMax+1];
long yMax[_counterMax+1];
long _endKeeper[_counterMax+1];
int tempCount=0;
for (int i = 0; i < Q_len + 1; i++){
for (int j = 0; j < D_len + 1; j++){
if(_MAX_SIMILARITY == ScoreTable[i][j]){
_endKeeper[tempCount] = j;
xMax[tempCount] = i;
yMax[tempCount] = j;
tempCount++;
}
}
}
double traceTimeInit = gettime();
for(int i = 0; i< _counterMax+1; i++){
long currXpos = xMax[i];
long currYpos = yMax[i];
char xElem = Q[currXpos];
char yElem = D[currYpos];
char currNode = ScoreTable[currXpos][currYpos];
char *_qOut = NULL;
_qOut = (char *)malloc((xMax[i]+1)*(yMax[i]+1)*sizeof(char));
if(_qOut==NULL){
printf("Error occured while trying to allocate memory for traceback.Terminating....");
exit(-1);
}
char *_dOut = NULL;
_dOut = (char *)malloc((xMax[i]+1)*(yMax[i]+1)* sizeof(char));
if(_dOut==NULL){
printf("Error occured while trying to allocate memory for traceback.Terminating....");
exit(-1);
}
int lengthCount = 0;
u_int8_t traceFlag = 0;
while(traceFlag != 1){
int up, left, diag;
up = ScoreTable[currXpos-1][currYpos];  
left = ScoreTable[currXpos][currYpos-1]; 
diag = ScoreTable[currXpos - 1][currYpos - 1];
if(diag == 0 && ScoreTable[currXpos][currYpos] - MATCH == 0){
traceFlag = 1;
currYpos--;
currXpos--;
xElem = Q[currXpos];
yElem = D[currYpos];
}else{
if( up > left && up > diag){
currXpos--;
xElem = Q[currXpos];
yElem = '-';
}else if (left > up && left > diag){
currYpos--;
xElem = '-';
yElem = D[currYpos];
}else if(diag >= up && diag >= left){
if(Q[currXpos-1] == D[currYpos-1] || (diag > up && diag > left) ){
currYpos--;
currXpos--;
xElem = Q[currXpos];
yElem = D[currYpos];
}else if(Q[currXpos-1] == D[currYpos]){
currXpos--;
xElem = Q[currXpos];
yElem = '-';
}else{
currYpos--;
xElem = '-';
yElem = D[currYpos];
}
}else{
if(Q[currXpos] == D[currYpos-1]){
currYpos--;
xElem = '-';
yElem = D[currYpos];
}else{
currXpos--;
xElem = Q[currXpos];
yElem = '-';
}
}
}
_traceSteps++;
_qOut[lengthCount] = xElem;
_dOut[lengthCount] = yElem;
lengthCount++;
}
_qOut[lengthCount] = '\0';
_dOut[lengthCount] = '\0';
_dOut = reverseArr(_dOut, strlen(_dOut));
_qOut = reverseArr(_qOut, strlen(_qOut));
fprintf(finFile, "\nMATCH %d [SCORE: %d,START: %ld,STOP: %ld]\n\tD: %s\n\tQ: %s\n", i+1, _MAX_SIMILARITY, (_endKeeper[i]-lengthCount) , _endKeeper[i]-1, _dOut, _qOut);
free(_qOut);
free(_dOut);
}
double traceTimeFin = gettime();
_totalTraceTime+=(traceTimeFin - traceTimeInit);
for(int i = 0; i< Q_len+1; i++) {
free(ScoreTable[i]);
}
}
long fillDataBuffer(char * buf, long bytereader ,int compFlag){
long index = 0;
long qIndex = 0;
long dIndex = 0;
Q = (char *)malloc(bytereader* sizeof(char));
if(Q==NULL){
printf("Error occured while trying to allocate memory for buffer.Terminating....");
exit(-1);
}
D = (char *)malloc(bytereader*sizeof(char));
if(D==NULL){
printf("Error occured while trying to allocate memory for buffer.Terminating....");
exit(-1);
}
uint8_t dFlag=0;
uint8_t qFlag=0;
uint8_t qCount=0;
size_t bufLen = strlen(buf);
if(compFlag == 1) return -1;
for(index;index < bufLen; index++){
if(buf[index] == ':' || buf[index] == '\\' || buf[index] == '\t'|| buf[index] == '\n' || buf[index] == '\r'){
continue;
}
if ( buf[index] == 'Q' ){
qFlag = 1;
qCount++;
if(qCount>1){
Q[qIndex] = '\0';
D[dIndex] = '\0';
dataParser();
free(Q);
free(D);
return index;
}
continue;
}else if(buf[index] == 'D'){
qFlag = 0;
dFlag = 1;
continue;
}
if(qFlag == 1){
Q[qIndex] = buf[index];
qIndex++;
}else if(dFlag == 1){
D[dIndex] = buf[index];
dIndex++;
}
}
Q[qIndex] = '\0';
D[dIndex] = '\0';
dataParser();
free(Q);
free(D);
return bufLen;
}
void fileParser(FILE *fp){
int BUFSIZE =  dSize+ qMax + 10000;
long numbytes = 0;
char buf[BUFSIZE];
int  sizeLeftover=0;
int isCompleted = 0;
long pos = 0;
int pairCounter = 0;
do
{
numbytes = fread(buf+sizeLeftover, 1, sizeof(buf)-1-sizeLeftover, fp);
if (numbytes<1)
{
isCompleted = 1;
numbytes  = 0;
}
buf[numbytes+sizeLeftover] = 0;
pos = fillDataBuffer(buf, numbytes+sizeLeftover,isCompleted);
pairCounter++;
if (pos<1)
{
isCompleted = 1;
pos      = 0;
}
sizeLeftover = numbytes+sizeLeftover-pos;
if (sizeLeftover<1) sizeLeftover=0;
if (pos!=0 && sizeLeftover!=0)
memmove(buf, buf+pos, sizeLeftover);
} while(!isCompleted && pairCounter != pairs);
}
int main(int argc, char * argv[]) {
double timeInitTotal = gettime();
commandChecker(argc, argv);
finFileDir();
FILE *fp;
fp = fopen(INPUT,"r");
if(fp == NULL){
printf("Error opening file\n");
exit(-9);
}
finFile = fopen(cwd,"a");
if(finFile == NULL){
printf("Error while opening write file!\n");
exit(1);
}
fileHeaderValues(fp);
fileParser(fp);
fclose(fp);
fclose(finFile);
double timeFinTotal = gettime();
_totalTime = timeFinTotal - timeInitTotal;
terminalPrinter();
return 0;
}
