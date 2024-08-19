int main(int argc, char *argv[])
{
char *auto_ = NULL;
#pragma omp task auto(auto_[1;-2])
{
}
#pragma omp taskwait
return 0;
}
