#include "bare_concurrent_set.h"

#include <gtest/gtest.h>
#include <string>
#include <unordered_set>
#include "reducer.h"

TEST(BareConcurrentSetTest, Initialization) {
hpmr::BareConcurrentSet<std::string> m;
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(BareConcurrentSetTest, Reserve) {
hpmr::BareConcurrentSet<std::string> m;
m.reserve(1000);
EXPECT_GE(m.get_n_buckets(), 1000);
}

TEST(BareConcurrentSetTest, CopyConstructor) {
hpmr::BareConcurrentSet<std::string> m;
std::hash<std::string> hasher;
m.set("aa", hasher("aa"));
EXPECT_TRUE(m.has("aa", hasher("aa")));
m.set("bb", hasher("bb"));
EXPECT_TRUE(m.has("bb", hasher("bb")));

hpmr::BareConcurrentSet<std::string> m2(m);
EXPECT_TRUE(m2.has("aa", hasher("aa")));
EXPECT_TRUE(m2.has("bb", hasher("bb")));
}

TEST(BareConcurrentSetTest, LargeReserve) {
hpmr::BareConcurrentSet<std::string> m;
const size_t LARGE_N_BUCKETS = 1000000;
m.reserve(LARGE_N_BUCKETS);
const size_t n_buckets = m.get_n_buckets();
EXPECT_GE(n_buckets, LARGE_N_BUCKETS);
}

TEST(BareConcurrentSetTest, GetAndSetLoadFactor) {
hpmr::BareConcurrentSet<int> m;
constexpr int N_KEYS = 100;
m.set_max_load_factor(0.5);
EXPECT_EQ(m.get_max_load_factor(), 0.5);
std::hash<int> hasher;
for (int i = 0; i < N_KEYS; i++) {
m.set(i, hasher(i));
}
EXPECT_GE(m.get_n_buckets(), N_KEYS / 0.5);
}

TEST(BareConcurrentSetTest, SetAndGet) {
hpmr::BareConcurrentSet<std::string> m;
std::hash<std::string> hasher;
m.set("aa", hasher("aa"));
EXPECT_TRUE(m.has("aa", hasher("aa")));
m.set("aa", hasher("aa"));
EXPECT_TRUE(m.has("aa", hasher("aa")));
m.set("cc", hasher("cc"));
EXPECT_TRUE(m.has("cc", hasher("cc")));
}

TEST(BareConcurrentSetTest, LargeParallelSetIndependentSTLComparison) {
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

TEST(BareConcurrentSetTest, LargeParallelSet) {
hpmr::BareConcurrentSet<std::string> m;
std::hash<std::string> hasher;
constexpr int N_KEYS = 1000000;
m.reserve(N_KEYS);
#pragma omp parallel for
for (int i = 0; i < N_KEYS; i++) {
const auto& key = std::to_string(i);
m.set(key, hasher(key));
}
EXPECT_EQ(m.get_n_keys(), N_KEYS);
EXPECT_GE(m.get_n_buckets(), N_KEYS);
}

TEST(BareConcurrentSetTest, LargeParallelAsyncSet) {
hpmr::BareConcurrentSet<std::string> m;
std::hash<std::string> hasher;
constexpr int N_KEYS = 1000000;
m.reserve(N_KEYS);
#pragma omp parallel for
for (int i = 0; i < N_KEYS; i++) {
const auto& key = std::to_string(i);
m.async_set(key, hasher(key));
}
m.sync();
EXPECT_EQ(m.get_n_keys(), N_KEYS);
EXPECT_GE(m.get_n_buckets(), N_KEYS);
}

TEST(BareConcurrentSetTest, UnsetAndHas) {
hpmr::BareConcurrentSet<std::string> m;
std::hash<std::string> hasher;
m.set("aa", hasher("aa"));
m.set("bbb", hasher("bbb"));
EXPECT_TRUE(m.has("aa", hasher("aa")));
EXPECT_TRUE(m.has("bbb", hasher("bbb")));
m.unset("aa", hasher("aa"));
EXPECT_FALSE(m.has("aa", hasher("aa")));
EXPECT_EQ(m.get_n_keys(), 1);

m.unset("not_exist_key", hasher("not_exist_key"));
EXPECT_EQ(m.get_n_keys(), 1);

m.unset("bbb", hasher("bbb"));
EXPECT_FALSE(m.has("aa", hasher("aa")));
EXPECT_FALSE(m.has("bbb", hasher("bbb")));
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(BareConcurrentSetTest, Clear) {
hpmr::BareConcurrentSet<std::string> m;
std::hash<std::string> hasher;
m.set("aa", hasher("aa"));
m.set("bbb", hasher("bbb"));
EXPECT_EQ(m.get_n_keys(), 2);
m.clear();
EXPECT_EQ(m.get_n_keys(), 0);
}

TEST(BareConcurrentSetTest, ClearAndShrink) {
hpmr::BareConcurrentSet<int> m;
std::hash<int> hasher;
constexpr int N_KEYS = 1000000;
#pragma omp parallel for schedule(static, 1)
for (int i = 0; i < N_KEYS; i++) {
m.set(i, hasher(i));
}
EXPECT_EQ(m.get_n_keys(), N_KEYS);
EXPECT_GE(m.get_n_buckets(), N_KEYS * m.get_max_load_factor());
m.clear_and_shrink();
EXPECT_EQ(m.get_n_keys(), 0);
EXPECT_LT(m.get_n_buckets(), N_KEYS * m.get_max_load_factor());
}

TEST(BareConcurrentSetTest, ToAndFromString) {
hpmr::BareConcurrentSet<std::string> m1;
std::hash<std::string> hasher;
m1.set("aa", hasher("aa"));
m1.set("bbb", hasher("bbb"));
const std::string serialized = m1.to_string();
hpmr::BareConcurrentSet<std::string> m2;
m2.from_string(serialized);
EXPECT_EQ(m2.get_n_keys(), 2);
EXPECT_TRUE(m2.has("aa", hasher("aa")));
EXPECT_TRUE(m2.has("bbb", hasher("bbb")));
}
