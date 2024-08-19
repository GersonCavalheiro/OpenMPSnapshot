

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_categories.h>


namespace hydra_thrust
{


struct input_universal_iterator_tag
{
operator input_host_iterator_tag () {return input_host_iterator_tag();}

operator input_device_iterator_tag () {return input_device_iterator_tag();}
};

struct output_universal_iterator_tag
{
operator output_host_iterator_tag () {return output_host_iterator_tag();}

operator output_device_iterator_tag () {return output_device_iterator_tag();}
};

struct forward_universal_iterator_tag
: input_universal_iterator_tag
{
operator forward_host_iterator_tag () {return forward_host_iterator_tag();};

operator forward_device_iterator_tag () {return forward_device_iterator_tag();};
};

struct bidirectional_universal_iterator_tag
: forward_universal_iterator_tag
{
operator bidirectional_host_iterator_tag () {return bidirectional_host_iterator_tag();};

operator bidirectional_device_iterator_tag () {return bidirectional_device_iterator_tag();};
};


namespace detail
{

template<typename T>
struct one_degree_of_separation
: T
{
};

} 


struct random_access_universal_iterator_tag
{
operator random_access_host_iterator_tag () {return random_access_host_iterator_tag();};

operator random_access_device_iterator_tag () {return random_access_device_iterator_tag();};

operator detail::one_degree_of_separation<bidirectional_universal_iterator_tag> () {return detail::one_degree_of_separation<bidirectional_universal_iterator_tag>();}

};


} 

