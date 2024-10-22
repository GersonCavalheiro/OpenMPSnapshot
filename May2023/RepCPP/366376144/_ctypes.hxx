#pragma once
#include <istream>
#include <ostream>

using std::istream;
using std::ostream;





#ifndef NONE
struct None {
friend bool operator==(None l, None r)     noexcept { return true; }
template <class T>
friend bool operator==(None l, const T& r) noexcept { return false; }
template <class T>
friend bool operator==(const T& l, None r) noexcept { return false; }


friend istream& operator>>(istream& a, None& x) noexcept { return a; }
friend ostream& operator<<(ostream& a, None x)  noexcept { return a; }
};
#define NONE None
#endif





#ifndef tclass0
#define tclass0 template <> class
#define tclass1 template <class> class
#define tclass2 template <class, class> class
#define tclass3 template <class, class, class> class
#define tclass0s template <class...> class
#define tclass1s template <class, class...> class
#define tclass2s template <class, class, class...> class
#define tclass3s template <class, class, class, class...> class
#endif
