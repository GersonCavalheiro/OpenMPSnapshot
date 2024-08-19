int main(int argc, char *argv[])
{
#pragma omp parallel for
for (int i(0) ; 
i < 10 ; 
i++) 
;
return 0;
}
