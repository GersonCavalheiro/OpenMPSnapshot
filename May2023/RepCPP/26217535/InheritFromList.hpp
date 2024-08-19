

#pragma once

namespace alpaka::meta
{
template<typename TBaseList>
class InheritFromList;

template<template<typename...> class TList, typename... TBases>
class InheritFromList<TList<TBases...>> : public TBases...
{
};
} 
