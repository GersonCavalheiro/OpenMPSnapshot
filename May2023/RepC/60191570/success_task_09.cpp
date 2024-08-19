#pragma omp task priority( iteration )
void task( int iteration, int number )
{
}
int main( void )
{
int i = 0;
for ( i = 20; i > 0; --i ) {
task( i, 1 );
task( i, 2 );
task( i, 3 );
}
#pragma omp taskwait
return 0;
}
