

#ifndef LBT_TESTING_UTILITIES
#define LBT_TESTING_UTILITIES
#pragma once

#include <type_traits>

#include <gtest/gtest.h>

#include "general/tuple_utilities.hpp"


namespace lbt {
namespace test {


template <class T,
typename std::enable_if_t<lbt::is_tuple_v<T>>* = nullptr>
class ToTestingTypes {
protected:
ToTestingTypes() = delete;
ToTestingTypes(ToTestingTypes const&) = delete;
ToTestingTypes(ToTestingTypes&&) = delete;
ToTestingTypes& operator=(ToTestingTypes const&) = delete;
ToTestingTypes& operator=(ToTestingTypes&&) = delete;


template <typename... Ts>
static constexpr auto helper(std::tuple<Ts...>&&) noexcept {
return ::testing::Types<Ts...>{};
}

public:
using type = decltype(helper(std::declval<T>()));
};
template<typename T>
using ToTestingTypes_t = typename ToTestingTypes<T>::type;



template <typename C>
struct TemplateDataType {
protected:
TemplateDataType() = delete;
TemplateDataType(TemplateDataType const&) = delete;
TemplateDataType(TemplateDataType&&) = delete;
TemplateDataType& operator=(TemplateDataType const&) = delete;
TemplateDataType& operator=(TemplateDataType&&) = delete;


template <template <typename T> class TC, typename T>
static constexpr auto helper(TC<T> const&) -> std::decay_t<decltype(std::declval<T>())>;
public:
using type = decltype(helper(std::declval<C>()));
};
template <typename C>
using TemplateDataType_t = typename TemplateDataType<C>::type;

}
}

#endif 
