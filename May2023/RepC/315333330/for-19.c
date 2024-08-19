const char *p = "abcde";
template <typename T>
void
f1 (void)
{
#pragma omp for
for (void *q = (void *)p; q < (void *) (p + 4); q++)	
;
}
template <typename T>
void
f2 (void)
{
#pragma omp for
for (const char *q = p; q < p + 4; q++)
;
}
template <typename T>
void
f3 (void)
{
#pragma omp for
for (T q = T (p); q < T (p + 4); q++)	
;
}
int
main (void)
{
f1 <int> ();		
f2 <int> ();
f3 <const char *> ();
f3 <void *> ();	
}
