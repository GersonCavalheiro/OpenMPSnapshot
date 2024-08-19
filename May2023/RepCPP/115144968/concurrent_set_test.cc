#include "concurrent_set.h"

#include <gtest/gtest.h>
#include <string>
#include <unordered_set>
#include "reducer.h"

TEST(ConcurrentSetTest, Initialization) {
hpmr::ConcurrentSet<std::string> m;
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(ConcurrentSetTest, Reserve) {
hpmr::ConcurrentSet<std::string> m;
m.reserve(1000);
EXPECT_GE(m.get_n_buckets(), 1000);
}

TEST(ConcurrentSetTest, CopyConstructor) {
hpmr::ConcurrentSet<std::string> m;
m.set("aa");
EXPECT_TRUE(m.has("aa"));
m.set("bb");
EXPECT_TRUE(m.has("bb"));

hpmr::ConcurrentSet<std::string> m2(m);
EXPECT_TRUE(m2.has("aa"));
EXPECT_TRUE(m2.has("bb"));
}

TEST(ConcurrentSetTest, LargeReserve) {
hpmr::ConcurrentSet<std::string> m;
const size_t LARGE_N_BUCKETS = 1000000;
m.reserve(LARGE_N_BUCKETS);
const size_t n_buckets = m.get_n_buckets();
EXPECT_GE(n_buckets, LARGE_N_BUCKETS);
}

TEST(ConcurrentSetTest, GetAndSetLoadFactor) {
hpmr::ConcurrentSet<int> m;
constexpr int N_KEYS = 100;
m.set_max_load_factor(0.5);
EXPECT_EQ(m.get_max_load_factor(), 0.5);
for (int i = 0; i < N_KEYS; i++) {
m.set(i);
}
EXPECT_GE(m.get_n_buckets(), N_KEYS / 0.5);
}

TEST(ConcurrentSetTest, SetAndGet) {
hpmr::ConcurrentSet<std::string> m;
m.set("aa");
EXPECT_TRUE(m.has("aa"));
m.set("aa");
EXPECT_TRUE(m.has("aa"));
m.set("cc");
EXPECT_TRUE(m.has("cc"));
}

TEST(ConcurrentSetTest, LargeParallelSetIndependentSTLComparison) {
const int n_threads = omp_get_max_threads();
std::unordered_set<std::string> m[n_threads];
constexpr int N_KEYS = 1000000;
for (int i = 0; i < n_threads; i++) m[i].reserve(N_KEYS / n_threads);
#pragma omp parallel for
for (int i = 0; i < N_KEYS; i++) {
const auto& key = std::to_string(i);
const int thread_id = omp_get_thread_num();
m[thread_id].insert(key);
}
}

TEST(ConcurrentSetTest, LargeParallelSet) {
hpmr::ConcurrentSet<std::string> m;
constexpr int N_KEYS = 1000000;
m.reserve(N_KEYS);
#pragma omp parallel for
for (int i = 0; i < N_KEYS; i++) {
const auto& key = std::to_string(i);
m.set(key);
}
EXPECT_EQ(m.get_n_keys(), N_KEYS);
EXPECT_GE(m.get_n_buckets(), N_KEYS);
}

TEST(ConcurrentSetTest, LargeParallelAsyncSet) {
hpmr::ConcurrentSet<std::string> m;
constexpr int N_KEYS = 1000000;
m.reserve(N_KEYS);
#pragma omp parallel for
for (int i = 0; i < N_KEYS; i++) {
const auto& key = std::to_string(i);
m.async_set(key);
}
m.sync();
EXPECT_EQ(m.get_n_keys(), N_KEYS);
EXPECT_GE(m.get_n_buckets(), N_KEYS);
}

TEST(ConcurrentSetTest, UnsetAndHas) {
hpmr::ConcurrentSet<std::string> m;
m.set("aa");
m.set("bbb");
EXPECT_TRUE(m.has("aa"));
EXPECT_TRUE(m.has("bbb"));
m.unset("aa");
EXPECT_FALSE(m.has("aa"));
EXPECT_EQ(m.get_n_keys(), 1);

m.unset("not_exist_key");
EXPECT_EQ(m.get_n_keys(), 1);

m.unset("bbb");
EXPECT_FALSE(m.has("aa"));
EXPECT_FALSE(m.has("bbb"));
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(ConcurrentSetTest, Clear) {
hpmr::ConcurrentSet<std::string> m;
m.set("aa");
m.set("bbb");
EXPECT_EQ(m.get_n_keys(), 2);
m.clear();
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(ConcurrentSetTest, ClearAndShrink) {
hpmr::ConcurrentSet<int> m;
constexpr int N_KEYS = 1000000;
#pragma omp parallel for schedule(static, 1)
for (int i = 0; i < N_KEYS; i++) {
m.set(i);
}
EXPECT_EQ(m.get_n_keys(), N_KEYS);
EXPECT_GE(m.get_n_buckets(), N_KEYS * m.get_max_load_factor());
m.clear_and_shrink();
EXPECT_EQ(m.get_n_keys(), 0);
EXPECT_LT(m.get_n_buckets(), N_KEYS * m.get_max_load_factor());
}

TEST(ConcurrentSetTest, ToAndFromString) {
hpmr::ConcurrentSet<std::string> m1;
m1.set("aa");
m1.set("bbb");
const std::string serialized = m1.to_string();
hpmr::ConcurrentSet<std::string> m2;
m2.from_string(serialized);
EXPECT_EQ(m2.get_n_keys(), 2);
EXPECT_TRUE(m2.has("aa"));
EXPECT_TRUE(m2.has("bbb"));
}
