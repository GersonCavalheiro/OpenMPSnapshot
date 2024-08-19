

#ifndef C_EWISE_H_
#define C_EWISE_H_

namespace bc {
namespace benchmarks {

template<class scalar_t, class allocator=bc::Basic_Allocator<scalar_t>>
auto c_cwise(int size, int reps) {

using vec   = bc::Vector<scalar_t, allocator>;

vec a(size);
vec b(size);
vec c(size);
vec d(size);
vec e(size);

a.randomize(-1000, 1000);
b.randomize(-1000, 1000);
c.randomize(-1000, 1000);
d.randomize(-1000, 1000);
e.randomize(-1000, 1000);

auto* a_ = a.data();
auto* b_ = b.data();
auto* c_ = c.data();
auto* d_ = d.data();
auto* e_ = e.data();



auto f = [&]() {
for (int i = 0; i < reps; ++i) {
#pragma omp parallel for
for (int j = 0; j < size; ++j) {
a_[j] = b_[j] + c_[j] - d_[j] / e_[j];
}
#pragma omp barrier
}
};
return timeit(f, 1);

}


}
}




#endif 
