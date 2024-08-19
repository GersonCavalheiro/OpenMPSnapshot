#include<stdio.h>
#include<omp.h>
int main()
{
int i,value=10,fst_var=5,shr_var=10,lst_var=20,sum=0;
printf("Values outside parallel region\n");
printf("value=%d\tfst_var=%d\tshr_var=%d\tlst_var=%d\tsum=%d\n\n",value,fst_var,shr_var,lst_var,sum);
#pragma omp parallel num_threads(4) private(value) firstprivate(fst_var) shared(shr_var)
{
printf("Values inside parallel region\n");
printf("value(private)=%d\n",value);
printf("first-var(firstprivate)=%d\n",fst_var);
shr_var++;printf("shr_var(shared)=%d\n",shr_var);
#pragma omp for firstprivate(lst_var) lastprivate(lst_var) reduction(+:sum) ordered
for(i=0;i<4;i++)
{
sum=i*i;
printf("sum=%d\t",sum);
}
}
printf("\n\n");
printf("Values outside parallel region\n");
printf("value(private)=%d\n",value);
printf("lst_var(lastprivate)=%d\n",lst_var);
printf("final sum(reduction)=%d\n",sum);
return 0;
}
