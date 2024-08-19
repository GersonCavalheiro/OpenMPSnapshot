#pragma once
template <typename T>
inline
void fill_randomly(std::vector<T> &in) {
std::generate(in.begin(), in.end(), []() { return costa::random_generator<T>::sample();});
}

template <typename T>
inline
void fill_zeros(std::vector<T> &in) {
std::generate(in.begin(), in.end(), []() { return 0;});
}

template <typename T>
inline
bool validate_results(std::vector<T>& v1, std::vector<T>& v2, double epsilon=1e-6) {
if (v1.size() != v2.size())
return false;
if (v1.size() == 0)
return true;
bool correct = true;
for (size_t i = 0; i < v1.size(); ++i) {
if (std::abs(v1[i] - v2[i]) > epsilon) {
std::cout << "v1 = " << v1[i] << ", which is != " << v2[i] << std::endl;
correct = false;
}
}
return correct;
}

