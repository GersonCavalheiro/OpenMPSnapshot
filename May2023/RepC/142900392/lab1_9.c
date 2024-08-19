# include <stdio.h>
# include <omp.h>
# include<stdlib.h>
int isPrime(int n){
if(n<=1)
return 0;
if(n!=2 && n%2==0)
return 0;
int i;
for (i = 3; i * i<= n; i=i+2)
{
if(n%i==0)
return 0;
}
return 1;
}
int main(){
int i,j,n,upper,lower,count=0;
printf("==========EXERCISE 9==========\n");
printf("    PRIME NUMBER FINDER\n\t\tS. DHANYA ABHIRAMI\n\t\t16BCE0965\n");
printf("Enter size of list: ");
scanf("%d",&n);
printf("Enter lower limit: ");
scanf("%d",&lower);
printf("Enter upper limit: ");
scanf("%d",&upper);
int list[n];
printf("The generated list: ");
for (i = 0; i < n; i++) {
list[i]= (rand() % (upper - lower + 1)) + lower;
printf("%d ",list[i]);
}
printf("\n");
#pragma omp parallel for 
for(i=0;i<n;i++){
printf("Thread ID: %d\n",omp_get_thread_num() );        
if(isPrime(list[i])==1)
{
printf("%d: Prime\n",list[i] );
count=count+1 ;
}
}
printf("\nNumber of prime numbers: %d\n",count );
return(0);
}
