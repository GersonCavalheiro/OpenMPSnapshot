#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include "correlation.h"
#include "arrayIO.h"
#ifndef SIZE_MAX
#pragma message "using custom SIZE_MAX"
#define SIZE_MAX ((size_t)-1)
#endif
static void die(const char*);
static float* readInputVectorBin(FILE*,size_t*,size_t*);
static void die(const char* message) {
fprintf(stderr,"%s\n",message);
abort();
}
static float* readInputVectorBin(FILE* input,size_t* cols,size_t* rows) {
size_t out;
out=fread(cols,sizeof(size_t),1,input);
assert(out==1);
if(out!=1) {die("cannot read cols count from input");}
out=fread(rows,sizeof(size_t),1,input);
assert(out==1);
if(out!=1) {die("cannot read rows count from input");}
assert((SIZE_MAX/ *cols) > *rows );
assert((SIZE_MAX/ *rows) > *cols );
size_t all= *cols * *rows;
float* result=calloc(all,sizeof(float));
out=fread(result,sizeof(float),all,input);
assert(out==all);
if(out!=all) {die("cannot read all the input matrix, input shorter than expected");}
return result;
}
static void showUsage(char* exeName) {
fprintf(stderr, 
"Usage :\n"
"%s < inputFile.txt > output.bin\n"
"%s -i inputfile.txt -o outputFile.txt\n"
"%s -b < inputFile.bin > output.bin\n"
"%s -b -i inputFile.bin -o output.bin\n"
"arguments could be mixed with pipe style calling\n"
"-b flag is to put binary array as input\n"
, exeName,exeName,exeName,exeName);
exit(0);
}
int main(int argc,char** argv) {
FILE *input=stdin;
FILE *output=stdout;
char* inputPath=NULL;
char* outputPath=NULL;
bool binaryFlag=false;
int c;
opterr=0;
while((c = getopt(argc,argv,"i:o:hb"))!= -1) {
switch(c) {
case 'i':
inputPath=strdup(optarg);
input=fopen(inputPath,"r");
if(input==NULL) {
fprintf(stderr, "error reading [%s]\n", inputPath);
abort();
}
break;
case 'o':
outputPath=strdup(optarg);
output=fopen(outputPath,"w");
if(input==NULL) {
fprintf(stderr, "error writing [%s]\n", outputPath);
abort();
}
break;
case 'h':
showUsage(argv[0]);
break;
case 'b':
binaryFlag=true;
break;
default:
showUsage(argv[0]);
break;
}
}
if(isatty(fileno(output))||isatty(fileno(input))) {
showUsage(argv[0]);
}
fprintf(stderr,"reading input\n");
size_t cols=0;
size_t rows=0;
float* inputData;
if(binaryFlag) {
inputData=readInputVectorBin(input,&cols,&rows);
}else{
inputData=readInputVectors(input,&cols,&rows);
}
if(inputPath!=NULL) {
free(inputPath);
inputPath=NULL;
fclose(input);
}
fprintf(stderr,"read %zu cols ,%zu rows\n",cols,rows);
assert(rows>0);
assert(cols>0);
assert((SIZE_MAX/(rows-(size_t)1)) > (rows/(size_t)2));
float* outputData=getCorrelation(inputData,cols,rows);
free(inputData);
fprintf(stderr,"writing output data\n");
size_t total=rows/(size_t)2*(rows-(size_t)1);
fwrite(outputData,sizeof(float),total,output);
free(outputData);
if(outputPath!=NULL) {
free(outputPath);
outputPath=NULL;
fclose(output);
}
fprintf(stderr,"done\n");
return(EXIT_SUCCESS);
}
