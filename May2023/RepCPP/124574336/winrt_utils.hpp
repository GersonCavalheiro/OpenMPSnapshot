
#ifndef BOOST_ASIO_DETAIL_WINRT_UTILS_HPP
#define BOOST_ASIO_DETAIL_WINRT_UTILS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS_RUNTIME)

#include <codecvt>
#include <cstdlib>
#include <future>
#include <locale>
#include <robuffer.h>
#include <windows.storage.streams.h>
#include <wrl/implements.h>
#include <boost/asio/buffer.hpp>
#include <boost/system/error_code.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/socket_ops.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {
namespace winrt_utils {

inline Platform::String^ string(const char* from)
{
std::wstring tmp(from, from + std::strlen(from));
return ref new Platform::String(tmp.c_str());
}

inline Platform::String^ string(const std::string& from)
{
std::wstring tmp(from.begin(), from.end());
return ref new Platform::String(tmp.c_str());
}

inline std::string string(Platform::String^ from)
{
std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
return converter.to_bytes(from->Data());
}

inline Platform::String^ string(unsigned short from)
{
return string(std::to_string(from));
}

template <typename T>
inline Platform::String^ string(const T& from)
{
return string(from.to_string());
}

inline int integer(Platform::String^ from)
{
return _wtoi(from->Data());
}

template <typename T>
inline Windows::Networking::HostName^ host_name(const T& from)
{
return ref new Windows::Networking::HostName((string)(from));
}

template <typename ConstBufferSequence>
inline Windows::Storage::Streams::IBuffer^ buffer_dup(
const ConstBufferSequence& buffers)
{
using Microsoft::WRL::ComPtr;
using boost::asio::buffer_size;
std::size_t size = buffer_size(buffers);
auto b = ref new Windows::Storage::Streams::Buffer(size);
ComPtr<IInspectable> insp = reinterpret_cast<IInspectable*>(b);
ComPtr<Windows::Storage::Streams::IBufferByteAccess> bacc;
insp.As(&bacc);
byte* bytes = nullptr;
bacc->Buffer(&bytes);
boost::asio::buffer_copy(boost::asio::buffer(bytes, size), buffers);
b->Length = size;
return b;
}

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
