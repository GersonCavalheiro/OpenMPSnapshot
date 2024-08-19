class A
{
public:
int _x;
A(int x) : _x(x) {}
};
int main() {
A a(2);
#pragma omp task
{
a;
}
}
