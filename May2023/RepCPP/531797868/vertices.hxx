#pragma once
#include <type_traits>
#include <vector>
#include <unordered_map>
#include "_main.hxx"

using std::remove_reference_t;
using std::vector;
using std::unordered_map;





template <class G>
inline auto vertexKeys(const G& x) {
return copyVector(x.vertexKeys());
}





template <class G>
inline auto vertexValues(const G& x) {
return copyVector(x.vertexKeys());
}





template <class G, class KS, class FM>
inline auto vertexData(const G& x, const KS& ks, FM fm) {
using K = typename G::key_type;
using V = typename G::vertex_value_type;
using T = remove_reference_t<decltype(fm(K(), V()))>;
vector<T> a;
for (K u : ks)
a.push_back(fm(u, x.vertexValue(u)));
return a;
}
template <class G, class KS>
inline auto vertexData(const G& x, const KS& ks) {
auto fm = [](auto u, auto d) { return d; };
return vertexData(x, ks, fm);
}
template <class G>
inline auto vertexData(const G& x) {
return copyVector(x.vertexValues());
}





template <class G, class T>
inline auto createContainer(const G& x, const T& _) {
return vector<T>(x.span());
}
template <class G, class T>
inline auto createCompressedContainer(const G& x, const T& _) {
return vector<T>(x.order());
}





template <class G, class T, class KS>
inline void decompressContainerW(vector<T>& a, const G& x, const vector<T>& vs, const KS& ks) {
scatterValuesW(a, vs, ks);
}
template <class G, class T>
inline void decompressContainerW(vector<T>& a, const G& x, const vector<T>& vs) {
decompressContainerW(a, x, vs, x.vertexKeys());
}

template <class G, class T, class KS>
inline auto decompressContainer(const G& x, const vector<T>& vs, const KS& ks) {
auto a = createContainer(x, T());
decompressContainerW(a, x, vs, ks);
return a;
}
template <class G, class T>
inline auto decompressContainer(const G& x, const vector<T>& vs) {
return decompressContainer(x, vs, x.vertexKeys());
}


template <class G, class K>
inline void decompressKeyContainerW(vector<K>& a, const G& x, const vector<K>& vs, const vector<K>& ks) {
auto fm = [&](auto i) { return ks[i]; };
scatterValuesW(a, vs, ks, fm);
}
template <class G, class K>
inline void decompressKeyContainerW(vector<K>& a, const G& x, const vector<K>& vs) {
decompressKeyContainerW(a, x, vs, vertexKeys(x));
}

template <class G, class K>
inline auto decompressKeyContainer(const G& x, const vector<K>& vs, const vector<K>& ks) {
auto a = createContainer(x, K());
decompressKeyContainerW(a, x, vs, ks);
return a;
}
template <class G, class K>
inline auto decompressKeyContainer(const G& x, const vector<K>& vs) {
return decompressKeyContainer(x, vs, vertexKeys(x));
}





template <class G, class T, class KS>
inline void compressContainerW(vector<T>& a, const G& x, const vector<T>& vs, const KS& ks) {
gatherValuesW(a, vs, ks);
}
template <class G, class T>
inline void compressContainerW(vector<T>& a, const G& x, const vector<T>& vs) {
return compressContainerW(a, x, vs, x.vertexKeys());
}

template <class G, class T, class KS>
inline auto compressContainer(const G& x, const vector<T>& vs, const KS& ks) {
auto a = createCompressedContainer(x, T());
compressContainerW(a, x, vs, ks);
return a;
}
template <class G, class T>
inline auto compressContainer(const G& x, const vector<T>& vs) {
return compressContainer(x, vs, x.vertexKeys());
}


template <class G, class K, class KS>
inline void compressKeyContainerW(vector<K>& a, const G& x, const vector<K>& vs, const KS& ks) {
auto m  = valueIndicesUnorderedMap(ks);
auto fm = [&](auto k) { return m[k]; };
gatherValuesW(a, vs, ks, fm);
}
template <class G, class K>
inline void compressKeyContainerW(vector<K>& a, const G& x, const vector<K>& vs) {
return compressKeyContainerW(a, x, vs, x.vertexKeys());
}

template <class G, class K, class KS>
inline auto compressKeyContainer(const G& x, const vector<K>& vs, const KS& ks) {
auto a = createCompressedContainer(x, K());
compressKeyContainerW(a, x, vs, ks);
return a;
}
template <class G, class K>
inline auto compressKeyContainer(const G& x, const vector<K>& vs) {
return compressKeyContainer(x, vs, x.vertexKeys());
}
