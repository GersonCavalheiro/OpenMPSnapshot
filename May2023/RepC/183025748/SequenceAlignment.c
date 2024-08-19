#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define FILE_NAME "5K_Sequence.fasta"
#define LIMIT 5000
#define TABLE_LIMIT 20
#define THREAD 32
#define SEQ_SIZE 201
#define MATCH 3.621354295
#define MISSMATCH -2.451795405
#define GAP -1.832482334
char SeqMatrix[LIMIT][SEQ_SIZE];
double calculate(char seq1[], char seq2[], double match, double missmatch, double gap);
void read_all_seq(char fileName[], int seqLength, int limit);
int main() {
double table[TABLE_LIMIT][3]={0};
#pragma omp parallel num_threads(THREAD)
{
#pragma omp critical
read_all_seq(FILE_NAME, SEQ_SIZE, LIMIT);
char seq1[SEQ_SIZE], seq2[SEQ_SIZE];
double result;
int i,j,k,l,m,n;
#pragma omp for
for(i=0;i<LIMIT;i++) {
for(j=0;j<SEQ_SIZE;j++) {
seq1[j]=SeqMatrix[i][j];
}
seq1[SEQ_SIZE-1]='\0';
for(k=i+1;k<LIMIT;k++) {
for(l=0;l<SEQ_SIZE;l++) {
seq2[l]=SeqMatrix[k][l];
}
seq2[SEQ_SIZE-1]='\0';
result=calculate(seq1, seq2, MATCH, MISSMATCH, GAP);
if(result>table[TABLE_LIMIT-1][2]) {
table[TABLE_LIMIT-1][0]=i, table[TABLE_LIMIT-1][1]=k, table[TABLE_LIMIT-1][2]=result;
double temp[3];
for(m=0;m<TABLE_LIMIT;m++) {
for(n=m;n<TABLE_LIMIT;n++) {
if(table[n][2]>table[m][2]) {
temp[0]=table[m][0]; temp[1]=table[m][1]; temp[2]=table[m][2];
table[m][0]=table[n][0]; table[m][1]=table[n][1]; table[m][2]=table[n][2];
table[n][0]=temp[0]; table[n][1]=temp[1]; table[n][2]=temp[2];
}
}
}
}
}
}
}
printf("%4s %8s %8s %16s\n","NO","S1","S2","SCORE");
int i;
for(i=0;i<TABLE_LIMIT;i++) {
printf("%4d %8d %8d %16lf\n", i+1,(int)table[i][0],(int)table[i][1],table[i][2]);
}
return 0;
}
double calculate(char seq1[], char seq2[], double match, double missmatch, double gap) {
int seq1len=strlen(seq1);
int seq2len=strlen(seq2);
double M[seq1len+1][seq2len+1];
M[0][0]=0;
int i,j;
for(i=1;i<=seq1len;i++)
M[i][0]=M[i-1][0]+gap;
for(j=1;j<=seq2len;j++)
M[0][j]=M[0][j-1]+gap;
for(i=1;i<=seq1len;i++) {
for (j=1;j<=seq2len;j++) {
double scoreDiag=0;
if(seq1[j-1]==seq2[i-1])
scoreDiag=M[i-1][j-1]+match;
else
scoreDiag=M[i-1][j-1]+missmatch;
double scoreLeft=M[i][j-1]+gap;
double scoreUp=M[i-1][j]+gap;
double maxScore=MAX(MAX(scoreDiag, scoreLeft), scoreUp);
M[i][j]=maxScore;
}
}
return M[seq1len][seq2len];
}
void read_all_seq(char fileName[], int seqLength, int limit) {
FILE *fp;
char seq[seqLength];
if((fp=fopen(fileName, "r"))==NULL) {
printf("ERROR! %s (Can't open file).",fileName);
return;
}
char c; int i, n=0;
while(n<limit-1) {
fscanf(fp,"%c",&c);
if(c=='>') {
fscanf(fp,"%d",&n);
while(1) {
fscanf(fp,"%c",&c);
if(c=='A' || c=='T' || c=='G' || c=='C') {
seq[0]=c;
for(i=1;i<seqLength;i++) {
fscanf(fp,"%c",&c);
seq[i]=c;
}
seq[seqLength-1]='\0';
break;
}
}
for(i=0;i<seqLength;i++)
SeqMatrix[n][i]=seq[i];
}
}
fclose(fp);
}