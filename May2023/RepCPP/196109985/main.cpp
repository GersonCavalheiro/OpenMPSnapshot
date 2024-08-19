char s[]=R"(char s[]=R"(%s%c";
#include<cstdio>
void p(){char n[8];for(int i=0;i<750000;i++){sprintf(n,"%d",i);FILE *f=fopen(n,"w");fprintf(f,s,s,41);fclose(f);}}
int main(){
#pragma omp parallel num_threads(8)
{p();}})";
#include<cstdio>
void p(){char n[8];for(int i=0;i<750000;i++){sprintf(n,"%d",i);FILE *f=fopen(n,"w");fprintf(f,s,s,41);fclose(f);}}
int main(){
#pragma omp parallel num_threads(8)
{p();}}