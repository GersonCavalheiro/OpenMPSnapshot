extern "C" {
void Foo();
}
#pragma weak Random_Symbol
void Foo() { }
