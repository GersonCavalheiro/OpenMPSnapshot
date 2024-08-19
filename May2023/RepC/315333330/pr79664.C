constexpr int
f1 ()
{
int i = 0;
#pragma omp parallel for			
for (i = 0; i < 10; ++i)
;
return 0;
}
constexpr int
f2 ()
{
int i = 0;
#pragma omp parallel				
i = 5;
return 0;
}
constexpr int
f3 ()
{
int i = 0;
#pragma omp task				
i = 5;
return 0;
}
constexpr int
f4 ()
{
int i = 0;
#pragma omp for					
for (i = 0; i < 10; ++i)
;
return 0;
}
constexpr int
f5 ()
{
int i = 0;
#pragma omp taskloop				
for (i = 0; i < 10; ++i)
;
return 0;
}
constexpr int
f6 ()
{
int i = 0;
#pragma omp target teams			
i = 5;
return 0;
}
constexpr int
f7 ()
{
int i = 0;
#pragma omp target data map(tofrom:i)		
i = 5;
return 0;
}
constexpr int
f8 ()
{
int i = 0;
#pragma omp target				
i = 5;
return 0;
}
constexpr int
f9 ()
{
int i = 0;
#pragma omp sections				
{
#pragma omp section
i = 5;
}
return 0;
}
constexpr int
f10 ()
{
int i = 0;
#pragma omp ordered				
i = 1;
return 0;
}
constexpr int
f11 ()
{
int i = 0;
#pragma omp critical				
i = 1;
return 0;
}
constexpr int
f12 ()
{
int i = 0;
#pragma omp single				
i = 1;
return 0;
}
constexpr int
f13 ()
{
int i = 0;
#pragma omp master				
i = 1;
return 0;
}
constexpr int
f14 ()
{
int i = 0;
#pragma omp taskgroup				
i = 1;
return 0;
}
constexpr int
f15 ()
{
int i = 0;
#pragma omp target update to(i)			
i = 1;
return 0;
}
constexpr int
f16 ()
{
int i = 0;
#pragma omp target update to(i)			
return 0;
}
constexpr int
f17 ()
{
int i = 0;
#pragma omp target enter data map(to:i)		
return 0;
}
constexpr int
f18 ()
{
int i = 0;
#pragma omp target exit data map(from:i)	
return 0;
}
