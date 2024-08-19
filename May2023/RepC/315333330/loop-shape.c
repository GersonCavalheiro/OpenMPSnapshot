int main ()
{
int i;
int v = 32, w = 19;
int length = 1, num = 5;
int *abc;
#pragma acc kernels
#pragma acc loop gang worker vector
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(26)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(v)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(length: 16)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(length: v)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(16)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(v)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(num: 16)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(num: v)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(16)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(v)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(static: 16, num: 5)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(static: v, num: w)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(length)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(num)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(num, static: 6)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(static: 5, num)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(1, static:*)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(static:*, 1)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(1, static:*)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(num: 5, static: 4)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(num: v, static: w)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(num, static:num)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(length:length)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(num:length)
for (i = 0; i < 10; i++)
;  
#pragma acc kernels
#pragma acc loop worker(num:num)
for (i = 0; i < 10; i++)
;  
#pragma acc kernels
#pragma acc loop gang(16, 24) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(v, w) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(num: 1, num:2, num:3, 4) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(num: 1 num:2, num:3, 4) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(1, num:2, num:3, 4) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(num, num:5) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(length:num) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(5, length:length) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(num:length) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(length:5) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(1, num:2) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(static: * abc)
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(static:*num:1) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(num: 5 static: *) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(,static: *) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(,length:5) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(,num:10) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(,10) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(,10) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(,10) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(-12) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(-1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(num:-1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(num:1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(static:-1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop gang(static:1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(-1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(num:-1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop worker(num:1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(-1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(length:-1.0) 
for (i = 0; i < 10; i++)
;
#pragma acc kernels
#pragma acc loop vector(length:1.0) 
for (i = 0; i < 10; i++)
;
return 0;
}
