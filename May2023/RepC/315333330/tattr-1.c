__attribute__ ((target("arch=zEC12")))
void htm1(void)
{
__builtin_tend();
}
__attribute__ ((target("arch=z10")))
void htm0(void)
{
__builtin_tend(); 
}
void htmd(void)
{
__builtin_tend();
}
