int
main (void)
{
int i; 
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
{ int i; }
#pragma GCC diagnostic pop
}
