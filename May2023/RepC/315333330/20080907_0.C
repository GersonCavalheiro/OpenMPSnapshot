#pragma GCC diagnostic ignored "-Wreturn-type"
struct Foo { void func (); }; Foo & bar () { } struct Baz { Baz (Baz &); };
Baz dummy() { bar().func(); }
