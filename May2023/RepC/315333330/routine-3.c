#pragma acc routine gang
int
gang () 
{
#pragma acc loop gang worker vector
for (int i = 0; i < 10; i++)
{
}
return 1;
}
#pragma acc routine worker
int
worker () 
{
#pragma acc loop worker vector
for (int i = 0; i < 10; i++)
{
}
return 1;
}
#pragma acc routine vector
int
vector () 
{
#pragma acc loop vector
for (int i = 0; i < 10; i++)
{
}
return 1;
}
#pragma acc routine seq
int
seq ()
{
return 1;
}
int
main ()
{
int red = 0;
#pragma acc parallel copy (red)
{
#pragma acc loop reduction (+:red) 
for (int i = 0; i < 10; i++)
red += gang ();
#pragma acc loop reduction (+:red)
for (int i = 0; i < 10; i++)
red += worker ();
#pragma acc loop reduction (+:red)
for (int i = 0; i < 10; i++)
red += vector ();
#pragma acc loop gang reduction (+:red)  
for (int i = 0; i < 10; i++)
red += gang (); 
#pragma acc loop worker reduction (+:red)  
for (int i = 0; i < 10; i++)
red += gang (); 
#pragma acc loop vector reduction (+:red)  
for (int i = 0; i < 10; i++)
red += gang (); 
#pragma acc loop gang reduction (+:red)
for (int i = 0; i < 10; i++)
red += worker ();
#pragma acc loop worker reduction (+:red)  
for (int i = 0; i < 10; i++)
red += worker (); 
#pragma acc loop vector reduction (+:red)  
for (int i = 0; i < 10; i++)
red += worker (); 
#pragma acc loop gang reduction (+:red)
for (int i = 0; i < 10; i++)
red += vector ();
#pragma acc loop worker reduction (+:red)
for (int i = 0; i < 10; i++)
red += vector ();
#pragma acc loop vector reduction (+:red)  
for (int i = 0; i < 10; i++)
red += vector (); 
#pragma acc loop gang reduction (+:red)
for (int i = 0; i < 10; i++)
red += seq ();
#pragma acc loop worker reduction (+:red)
for (int i = 0; i < 10; i++)
red += seq ();
#pragma acc loop vector reduction (+:red)
for (int i = 0; i < 10; i++)
red += seq ();
}
return 0;
}
