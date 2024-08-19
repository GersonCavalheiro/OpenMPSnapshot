void f1 ()
{
#pragma acc kernels default 
;
#pragma acc parallel default 
;
#pragma acc kernels default ( 
;
#pragma acc parallel default ( 
;
#pragma acc kernels default (, 
;
#pragma acc parallel default (, 
;
#pragma acc kernels default () 
;
#pragma acc parallel default () 
;
#pragma acc kernels default (,) 
;
#pragma acc parallel default (,) 
;
#pragma acc kernels default (firstprivate) 
;
#pragma acc parallel default (firstprivate) 
;
#pragma acc kernels default (private) 
;
#pragma acc parallel default (private) 
;
#pragma acc kernels default (shared) 
;
#pragma acc parallel default (shared) 
;
#pragma acc kernels default (none 
;
#pragma acc parallel default (none 
;
#pragma acc kernels default (none none) 
;
#pragma acc parallel default (none none) 
;
#pragma acc kernels default (none, none) 
;
#pragma acc parallel default (none, none) 
;
}
