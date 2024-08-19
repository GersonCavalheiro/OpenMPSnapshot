#pragma acc update 
struct s_pragma_struct
{
#pragma acc update 
};
void
f_pragma_param (
#pragma acc update 
void)
{
}
void
f2 (void)
{
if (0)
#pragma acc update 
}
void
f3 (void)
{
int i = 0;
#pragma acc update device(i)
}
