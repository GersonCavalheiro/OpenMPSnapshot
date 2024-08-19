

#ifndef BOOST_BIMAP_SUPPORT_LAMBDA_HPP
#define BOOST_BIMAP_SUPPORT_LAMBDA_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/lambda/lambda.hpp>

namespace boost {
namespace bimaps {

namespace {



boost::lambda::placeholder1_type & _key  = boost::lambda::_1;
boost::lambda::placeholder1_type & _data = boost::lambda::_1;

}

} 
} 


#endif 

