#include "concurrent_map.h"

#include <gtest/gtest.h>
#include "reducer.h"

TEST(ConcurrentMapTest, Initialization) {
hpmr::ConcurrentMap<std::string, int> m;
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(ConcurrentMapTest, Reserve) {
hpmr::ConcurrentMap<std::string, int> m;
m.reserve(100);
EXPECT_GE(m.get_n_buckets(), 100);
}

TEST(ConcurrentMapTest, GetAndSetLoadFactor) {
hpmr::ConcurrentMap<int, int> m;
constexpr int N_KEYS = 100;
m.set_max_load_factor(0.5);
EXPECT_EQ(m.get_max_load_factor(), 0.5);
for (int i = 0; i < N_KEYS; i++) {
m.set(i, i);
}
EXPECT_GE(m.get_n_buckets(), N_KEYS / 0.5);
}

TEST(ConcurrentMapTest, ParallelInsertAndRehash) {
hpmr::ConcurrentMap<int, int> m;
constexpr int N_KEYS = 1000;
#pragma omp parallel for
for (int i = 0; i < N_KEYS; i++) {
m.set(i, i);
}
EXPECT_EQ(m.get_n_keys(), N_KEYS);
EXPECT_GE(m.get_n_buckets(), N_KEYS);
}

TEST(ConcurrentMapTest, LargeReserve) {
hpmr::ConcurrentMap<std::string, int> m;
const size_t LARGE_N_BUCKETS = 1000000;
m.reserve(LARGE_N_BUCKETS);
const size_t n_buckets = m.get_n_buckets();
EXPECT_GE(n_buckets, LARGE_N_BUCKETS);
}

TEST(ConcurrentMapTest, SetAndGet) {
hpmr::ConcurrentMap<std::string, int> m;
m.set("aa", 0);
EXPECT_EQ(m.get("aa"), 0);
m.set("aa", 1);
EXPECT_EQ(m.get("aa", 0), 1);
m.set("aa", 2, hpmr::Reducer<int>::sum);
EXPECT_EQ(m.get("aa"), 3);

hpmr::ConcurrentMap<std::string, int> m2;
m2.set("cc", 3, hpmr::Reducer<int>::sum);
EXPECT_EQ(m2.get("cc"), 3);
}

TEST(ConcurrentMapTest, AsyncSetAndGet) {
hpmr::ConcurrentMap<std::string, int> m;
m.async_set("aa", 0);
m.async_set("aa", 1);
m.async_set("bb", 2);
m.sync();
EXPECT_EQ(m.get("aa"), 1);
EXPECT_EQ(m.get("bb"), 2);
}

TEST(ConcurrentMapTest, Unset) {
hpmr::ConcurrentMap<std::string, int> m;
m.set("aa", 1);
m.set("bbb", 2);
m.unset("aa");
EXPECT_FALSE(m.has("aa"));
EXPECT_TRUE(m.has("bbb"));
EXPECT_EQ(m.get_n_keys(), 1);

m.unset("not_exist_key");
EXPECT_EQ(m.get_n_keys(), 1);

m.unset("bbb");
EXPECT_FALSE(m.has("aa"));
EXPECT_FALSE(m.has("bbb"));
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(ConcurrentMapTest, Clear) {
hpmr::ConcurrentMap<std::string, int> m;
m.set("aa", 1);
m.set("bbb", 2);
m.clear();
EXPECT_FALSE(m.has("aa"));
EXPECT_FALSE(m.has("bbb"));
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(ConcurrentMapTest, ClearAndShrink) {
hpmr::ConcurrentMap<int, int> m;
constexpr int N_KEYS = 100000;
for (int i = 0; i < N_KEYS; i++) {
m.set(i, i);
}
EXPECT_EQ(m.get_n_keys(), N_KEYS);
EXPECT_GE(m.get_n_buckets(), N_KEYS * m.get_max_load_factor());
m.clear_and_shrink();
EXPECT_EQ(m.get_n_keys(), 0);
EXPECT_LT(m.get_n_buckets(), N_KEYS * m.get_max_load_factor());
}
