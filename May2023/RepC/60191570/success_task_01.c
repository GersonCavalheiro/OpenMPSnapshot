#pragma oss task
void f()
{ }
void g() {
#pragma oss task
{
f();
}
#pragma oss task for chunksize(4)
for (int i = 0; i < 100; ++i)
{}
#pragma oss taskwait
}
