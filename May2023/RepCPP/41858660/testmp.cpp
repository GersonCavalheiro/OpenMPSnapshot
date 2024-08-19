
int main(int argc, const char * argv[])
{	
int w = 0;
#pragma omp parallel
{
#pragma omp task
if(argc == 10)
{
w = 100;
}
else
{
w = 20;
argc--;
}
}
while(w > 0)
{
if(w == 2)
break;
}
return w;
}