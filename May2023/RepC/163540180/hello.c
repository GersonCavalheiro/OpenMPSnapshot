int main(int argc, char** argv){

#pragma omp parallel
{
printf("Hello world.\n");
}

}
