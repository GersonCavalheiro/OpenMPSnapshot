#pragma once

namespace dg
{



struct AnyPolicyTag{};
struct NoPolicyTag{};

struct SerialTag    : public AnyPolicyTag{};
struct CudaTag      : public AnyPolicyTag{};
struct OmpTag       : public AnyPolicyTag{};


}
