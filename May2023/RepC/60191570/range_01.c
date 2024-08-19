int foo(int q)
{
int t = q;
int i;
int a = 5;
int c;
#pragma analysis_check assert range(a:5:5:0)
if (t < 10)
c = a;
else
c = a - 1;
#pragma analysis_check assert range(c:4:5:0)
for (i = 0; i < 4; ++i)
#pragma analysis_check assert range(i:0:3:0)
a = 10;
#pragma analysis_check assert range(i:4:4:0)
return a;
}
