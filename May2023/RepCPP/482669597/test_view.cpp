#include <gtest/gtest.h>
#include "Types.hpp"

void allocate_inside_function(RealView2D& reference_to_a_View, const int n, const int m) {
reference_to_a_View = RealView2D("simple", n, m);
#if defined( ENABLE_OPENMP_OFFLOAD )
#pragma omp target teams distribute parallel for simd collapse(2)
#endif
for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
reference_to_a_View(i, j) = i + j * 0.2 + 0.01;
}
}
}

void set_inside_function(RealView2D shallow_copy_to_a_View) {
const int n = shallow_copy_to_a_View.extent(0);
const int m = shallow_copy_to_a_View.extent(1);

#if defined( ENABLE_OPENMP_OFFLOAD )
#pragma omp target teams distribute parallel for simd collapse(2)
#endif
for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
shallow_copy_to_a_View(i, j) = i + j * 0.2 + 0.01;
}
}
}

TEST( VIEW, DEFAULT_CONSTRUCTOR ) {
RealView2D empty; 
RealView2D simple("simple", std::array<size_t, 2>{2, 3}); 
RealView2D Kokkos_like("kokkos_like", 2, 3); 
RealView2D offset_view("offset_view", std::array<size_t, 2>{3, 4}, std::array<int, 2>{-1, -1}); 
RealView2D offset_view_int("offset_view", std::array<int, 2>{3, 4}, std::array<int, 2>{-1, -1}); 
}

TEST( VIEW, COPY_CONSTRUCTOR ) {
RealView2D simple("simple", 16, 16);
RealView2D reference("reference", 16, 16);

set_inside_function(simple);

const int n = reference.extent(0);
const int m = reference.extent(1);

#if defined( ENABLE_OPENMP_OFFLOAD )
#pragma omp target teams distribute parallel for simd collapse(2)
#endif
for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
reference(i, j) = i + j * 0.2 + 0.01;
}
}

simple.updateSelf();
reference.updateSelf();

for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
ASSERT_EQ( simple(i, j), reference(i, j) );
ASSERT_NE( simple(i, j), 0.0 ); 
}
}
}

TEST( VIEW, ASSIGN ) {
RealView2D simple;
RealView2D reference("reference", 16, 16);

const int n = reference.extent(0);
const int m = reference.extent(1);

#if defined( ENABLE_OPENMP_OFFLOAD )
#pragma omp target teams distribute parallel for simd collapse(2)
#endif
for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
reference(i, j) = i + j * 0.2 + 0.01;
}
}

reference.updateSelf();
simple = reference;

for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
ASSERT_EQ( simple(i, j), reference(i, j) );
ASSERT_NE( simple(i, j), 0.0 ); 
}
}

simple.updateSelf();
for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
ASSERT_EQ( simple(i, j), reference(i, j) );
ASSERT_NE( simple(i, j), 0.0 ); 
}
}
}

TEST( VIEW, MOVE ) {
RealView2D simple;
RealView2D moved_reference("reference", 16, 16);
RealView2D reference("reference", 16, 16);

const int n = moved_reference.extent(0);
const int m = moved_reference.extent(1);

#if defined( ENABLE_OPENMP_OFFLOAD )
#pragma omp target teams distribute parallel for simd collapse(2)
#endif
for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
moved_reference(i, j) = i + j * 0.2 + 0.01;
reference(i, j) = moved_reference(i, j);
}
}

reference.updateSelf();
moved_reference.updateSelf();
simple = std::move(moved_reference);

for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
ASSERT_EQ( simple(i, j), reference(i, j) );
ASSERT_NE( simple(i, j), 0.0 ); 
}
}

simple.updateSelf();
for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
ASSERT_EQ( simple(i, j), reference(i, j) );
ASSERT_NE( simple(i, j), 0.0 ); 
}
}
}

TEST( VIEW, MOVE_ASSIGN ) {
RealView2D simple;
RealView2D reference("reference", 16, 16);

const int n = reference.extent(0);
const int m = reference.extent(1);

allocate_inside_function(simple, n, m);

#if defined( ENABLE_OPENMP_OFFLOAD )
#pragma omp target teams distribute parallel for simd collapse(2)
#endif
for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
reference(i, j) = i + j * 0.2 + 0.01;
}
}

simple.updateSelf();
reference.updateSelf();

for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
ASSERT_EQ( simple(i, j), reference(i, j) );
ASSERT_NE( simple(i, j), 0.0 ); 
}
}
}

TEST( VIEW, SWAP ) {

RealView2D a("a", 16, 16), b("b", 16, 16);
RealView2D a_ref("b", 16, 16), b_ref("a", 16, 16);

const int n = a.extent(0);
const int m = a.extent(1);

#if defined( ENABLE_OPENMP_OFFLOAD )
#pragma omp target teams distribute parallel for simd collapse(2)
#endif
for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
a(i, j) = i + j * 0.2 + 0.01;
b(i, j) = i *0.3 + j * 0.5 + 0.01;

a_ref(i, j) = b(i, j);
b_ref(i, j) = a(i, j);
}
}

a.updateSelf();
b.updateSelf();
a_ref.updateSelf();
b_ref.updateSelf();

a.swap(b);

ASSERT_EQ( a.name(), a_ref.name() );
ASSERT_EQ( b.name(), b_ref.name() );

for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
ASSERT_EQ( a(i, j), a_ref(i, j) );
ASSERT_EQ( b(i, j), b_ref(i, j) );
ASSERT_NE( a(i, j), 0.0 );
ASSERT_NE( b(i, j), 0.0 );
}
}

a.updateSelf();
b.updateSelf();
for(int j=0; j<m; j++) {
for(int i=0; i<n; i++) {
ASSERT_EQ( a(i, j), a_ref(i, j) );
ASSERT_EQ( b(i, j), b_ref(i, j) );
ASSERT_NE( a(i, j), 0.0 );
ASSERT_NE( b(i, j), 0.0 );
}
}
}
