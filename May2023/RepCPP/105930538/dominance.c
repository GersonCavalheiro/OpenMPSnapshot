

# include <stdio.h>
# include <stdlib.h>
# include <math.h>

# include "global.h"
# include "rand.h"




int check_dominance (individual *a, individual *b)
{
int i;
int flag1;
int flag2;
flag1 = 0;
flag2 = 0;

if (a->constr_violation<0 && b->constr_violation<0)
{
if (a->constr_violation > b->constr_violation)
{
return (1);
}
else
{
if (a->constr_violation < b->constr_violation)
{
return (-1);
}
else
{
return (0);
}
}
}
else
{
if (a->constr_violation < 0 && b->constr_violation == 0)
{
return (-1);
}
else
{
if (a->constr_violation == 0 && b->constr_violation <0)
{
return (1);
}
else
{

for (i = 0; i < nobj; i++)
{
if (a->obj[i] < b->obj[i])
{

flag1 = 1;

}
else
{
if(a->obj[i] > b->obj[i])
{

flag2 = 1;
}
}
}

if (flag1==1 && flag2==0)
{
return (1);
}
else
{
if (flag1 == 0 && flag2==1)
{
return (-1);
}
else
{
return (0);
}
}
}
}
}
}
