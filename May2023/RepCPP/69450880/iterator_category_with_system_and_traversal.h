

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

namespace hydra_thrust
{
namespace detail
{


template<typename Category, typename System, typename Traversal>
struct iterator_category_with_system_and_traversal
: Category
{
}; 


template<typename Category> struct iterator_category_to_system;

template<typename Category, typename System, typename Traversal>
struct iterator_category_to_system<iterator_category_with_system_and_traversal<Category,System,Traversal> >
{
typedef System type;
}; 


template<typename Category> struct iterator_category_to_traversal;

template<typename Category, typename System, typename Traversal>
struct iterator_category_to_traversal<iterator_category_with_system_and_traversal<Category,System,Traversal> >
{
typedef Traversal type;
}; 



} 
} 

