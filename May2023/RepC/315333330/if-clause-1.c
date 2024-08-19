void
f (void)
{
struct { int i; } *p;
#pragma acc data copyout(p) if(1) if(1) 
;
#pragma acc update device(p) if(*p) 
}
