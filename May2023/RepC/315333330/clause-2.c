struct A { int a; };
struct B { B(); };
struct C { C(); C(const C&); };
struct D { D& operator=(const D&); };
class E { private: E(); public: E(int); };	
class F { private: F(const F&); public: F(); };	
class G { private: G& operator=(const G&); };	
void bar();
void foo()
{
A a; B b; C c; D d; E e(0); F f; G g;
#pragma omp parallel shared(a, b, c, d, e, f, g)
bar();
#pragma omp parallel private(a, b, c, d, f, g)
bar();
#pragma omp parallel private(e)		
bar();
#pragma omp parallel firstprivate(a, b, c, d, e, g)
bar();
#pragma omp parallel firstprivate(f)		
bar();
#pragma omp parallel sections lastprivate(a, b, d, c, f)
{ bar(); }
#pragma omp parallel sections lastprivate(e)	
{ bar(); }
#pragma omp parallel sections lastprivate(g)	
{ bar(); }
#pragma omp parallel sections firstprivate(e) lastprivate(e)
{ bar(); }
}
