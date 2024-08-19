template<int>
void
f1 ()
{
#pragma omp parallel for
for (int i = ""; i < 4; ++i)	
;
}
template<int>
void
f2 ()
{
int i;
#pragma omp parallel for
for (i = ""; i < 4; ++i)	
;
}
template<typename T>
void
f3 ()
{
#pragma omp parallel for
for (T i = ""; i < 4; ++i)	
;
}
template<typename T>
void
f4 ()
{
T i;
#pragma omp parallel for
for (i = ""; i < 4; ++i)	
;
}
void
bar ()
{
f1<0> ();			
f2<1> ();			
f3<int> ();			
f4<int> ();			
}
