int main ()
{
#pragma acc parallel
{
#pragma acc loop tile (*,*)
for (int ix = 0; ix < 30; ix++)
; 
#pragma acc loop tile (*,*)
for (int ix = 0; ix < 30; ix++)
for (int jx = 0; jx < ix; jx++) 
;
#pragma acc loop tile (*)
for (int ix = 0; ix < 30; ix++)
for (int jx = 0; jx < ix; jx++) 
;
}
return 0;
}
