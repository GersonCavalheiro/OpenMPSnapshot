#pragma once

#include <cstdint> 
#include <tuple> 
#include <utility> 

namespace nlohmann
{


template<typename BinaryType>
class byte_container_with_subtype : public BinaryType
{
public:
using container_type = BinaryType;

byte_container_with_subtype() noexcept(noexcept(container_type()))
: container_type()
{}

byte_container_with_subtype(const container_type& b) noexcept(noexcept(container_type(b)))
: container_type(b)
{}

byte_container_with_subtype(container_type&& b) noexcept(noexcept(container_type(std::move(b))))
: container_type(std::move(b))
{}

byte_container_with_subtype(const container_type& b, std::uint8_t subtype) noexcept(noexcept(container_type(b)))
: container_type(b)
, m_subtype(subtype)
, m_has_subtype(true)
{}

byte_container_with_subtype(container_type&& b, std::uint8_t subtype) noexcept(noexcept(container_type(std::move(b))))
: container_type(std::move(b))
, m_subtype(subtype)
, m_has_subtype(true)
{}

bool operator==(const byte_container_with_subtype& rhs) const
{
return std::tie(static_cast<const BinaryType&>(*this), m_subtype, m_has_subtype) ==
std::tie(static_cast<const BinaryType&>(rhs), rhs.m_subtype, rhs.m_has_subtype);
}

bool operator!=(const byte_container_with_subtype& rhs) const
{
return !(rhs == *this);
}


void set_subtype(std::uint8_t subtype) noexcept
{
m_subtype = subtype;
m_has_subtype = true;
}


constexpr std::uint8_t subtype() const noexcept
{
return m_subtype;
}


constexpr bool has_subtype() const noexcept
{
return m_has_subtype;
}


void clear_subtype() noexcept
{
m_subtype = 0;
m_has_subtype = false;
}

private:
std::uint8_t m_subtype = 0;
bool m_has_subtype = false;
};

}  
