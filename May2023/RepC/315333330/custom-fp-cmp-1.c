#pragma GCC target ("custom-frdxhi=40")
#pragma GCC target ("custom-frdxlo=41")
#pragma GCC target ("custom-frdy=42")
#pragma GCC target ("custom-fwrx=43")
#pragma GCC target ("custom-fwry=44")
#pragma GCC target ("custom-fcmpeqs=200")
int
test_fcmpeqs (float a, float b)
{
return (a == b);
}
#pragma GCC target ("custom-fcmpgtd=201")
int
test_fcmpgtd (double a, double b)
{
return (a > b);
}
#pragma GCC target ("custom-fcmples=202")
int
test_fcmples (float a, float b)
{
return (a <= b);
}
#pragma GCC target ("custom-fcmpned=203")
int
test_fcmpned (double a, double b)
{
return (a != b);
}
