#include<stdio.h>
#include<omp.h>
int main() {
int n,i,m,j;
int a[19],b=0, sumofdigit=0;
omp_set_num_threads(2);
n=2021;
#pragma omp parallel
{
#pragma omp single
for(i=3;i<=n;i++)
{
int flag=0;
int final=i;
m=i/2;
for(j=2;j<=m;j++){ 
if(i%j==0){      
flag=1;    
break;    
}    
}    
if(flag==0){
int t=i;
int sum=0,r;
while(t>0)    
{    
r=t%10;    
sum=(sum*10)+r;    
t=t/10;    
}
if(final==sum){
printf("   [~] thread %d found %d\n",omp_get_thread_num(),final);
a[b]=final;
b++;
if(sum%2!=0){ 
int x;
while(sum>0)     
{    
x=sum%10;    
sumofdigit=sumofdigit+x;    
sum=sum/10;    
}    
}
else{continue;}
}    
}               	
}
#pragma omp barrier
#pragma omp single
{ 
printf("\n[+] the sum of digits of elements of prime palindromic series is %d as thread %d says.\n\n",sumofdigit,omp_get_thread_num());
for(i=0;i<19;i++)
printf("  [~]  %d -->says thread %d\n",a[i],omp_get_thread_num());
}
}
return 0;
}
