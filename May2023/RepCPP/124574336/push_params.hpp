

#ifndef BOOST_IOSTREAMS_DETAIL_PUSH_PARAMS_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_PUSH_PARAMS_HPP_INCLUDED 

#if defined(_MSC_VER)
# pragma once
#endif                    

#define BOOST_IOSTREAMS_PUSH_PARAMS() \
, std::streamsize buffer_size = -1 , std::streamsize pback_size = -1 \


#define BOOST_IOSTREAMS_PUSH_ARGS() , buffer_size, pback_size     

#endif 
