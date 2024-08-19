template <typename T>
void f ()
{
extern T n ();
#pragma omp parallel num_threads(n)	
;
#pragma omp parallel for schedule(static, n)
for (int i = 0; i < 10; i++)		
;
}
void g ()
{
f<int> ();
}
