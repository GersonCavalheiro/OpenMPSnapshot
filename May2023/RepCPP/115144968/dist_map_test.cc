#include "dist_map.h"

#include <gtest/gtest.h>
#include <string>
#include "reducer.h"

TEST(DistMapTest, Initialization) {
hpmr::DistMap<std::string, int> m;
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(DistMapTest, Reserve) {
hpmr::DistMap<std::string, int> m;
m.reserve(100);
EXPECT_GE(m.get_n_buckets(), 100);
}

TEST(DistMapTest, LargeReserve) {
hpmr::DistMap<std::string, int> m;
const size_t LARGE_N_BUCKETS = 1000000;
m.reserve(LARGE_N_BUCKETS);
const size_t n_buckets = m.get_n_buckets();
EXPECT_GE(n_buckets, LARGE_N_BUCKETS);
}

TEST(DistMapTest, SetAndGet) {
hpmr::DistMap<std::string, int> m;
m.async_set("aa", 1);
m.sync();
EXPECT_EQ(m.get("aa"), 1);
}

TEST(DistMapTest, GetAndSetLoadFactorAutoRehash) {
hpmr::DistMap<std::string, int> m;
constexpr int N_KEYS = 100000;
m.set_max_load_factor(0.5);
EXPECT_EQ(m.get_max_load_factor(), 0.5);
#pragma omp parallel for
for (int i = 0; i < N_KEYS; i++) {
m.async_set(std::to_string(i), i);
}
m.sync();
EXPECT_GE(m.get_n_buckets(), N_KEYS / 0.5);
}

TEST(DistMapTest, Clear) {
hpmr::DistMap<std::string, int> m;
m.async_set("aa", 1);
m.async_set("bbb", 2);
EXPECT_EQ(m.get_n_keys(), 2);
m.clear();
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(DistMapTest, ClearAndShrink) {
hpmr::DistMap<std::string, int> m;
constexpr int N_KEYS = 100000;
#pragma omp parallel for
for (int i = 0; i < N_KEYS; i++) {
m.async_set(std::to_string(i), i);
}
m.get_n_keys();
m.sync();
EXPECT_EQ(m.get_n_keys(), N_KEYS);
EXPECT_GE(m.get_n_buckets(), N_KEYS * m.get_max_load_factor());
m.clear_and_shrink();
EXPECT_EQ(m.get_n_keys(), 0);
EXPECT_LT(m.get_n_buckets(), N_KEYS * m.get_max_load_factor());
}

TEST(DistMapTest, LargeMapReduce) {
hpmr::DistMap<std::string, long long> m;
constexpr long long N_KEYS = 1000000;
m.reserve(N_KEYS);
#pragma omp parallel for
for (long long i = 0; i < N_KEYS; i++) {
m.async_set(std::to_string(i), i);
}
m.sync();
EXPECT_EQ(m.get_n_keys(), N_KEYS);
const auto& mapper = [](const std::string&,
const long long value,
const std::function<void(const int, const long long)>& emit) {
emit(0, value);
};
auto res = m.mapreduce<int, long long>(mapper, hpmr::Reducer<long long>::sum, true);
EXPECT_EQ(res.get(0), N_KEYS * (N_KEYS - 1) / 2);
}
