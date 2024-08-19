void
fun (void)
{
float *fp;
#pragma acc parallel copy(fp[0:2],fp[0:2]) 
;
#pragma acc kernels present_or_copyin(fp[3]) present_or_copyout(fp[7:4]) 
;
#pragma acc data create(fp[:10]) deviceptr(fp) 
;
#pragma acc data create(fp) present(fp) 
;
}
