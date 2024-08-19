

#pragma once

#include <aws/core/http/HttpTypes.h>
#include <utility>
#include <aws/core/http/HttpResponse.h>

namespace Aws
{

template <typename PAYLOAD_TYPE>
class AmazonWebServiceResult
{
public:
AmazonWebServiceResult() {}


AmazonWebServiceResult(const PAYLOAD_TYPE& payload, const Http::HeaderValueCollection& headers, Http::HttpResponseCode responseCode = Http::HttpResponseCode::OK) :
m_payload(payload),
m_responseHeaders(headers),
m_responseCode(responseCode)
{}


AmazonWebServiceResult(PAYLOAD_TYPE&& payload, Http::HeaderValueCollection&& headers, Http::HttpResponseCode responseCode = Http::HttpResponseCode::OK) :
m_payload(std::forward<PAYLOAD_TYPE>(payload)),
m_responseHeaders(std::forward<Http::HeaderValueCollection>(headers)),
m_responseCode(responseCode)
{}

AmazonWebServiceResult(const AmazonWebServiceResult& result) :
m_payload(result.m_payload),
m_responseHeaders(result.m_responseHeaders),
m_responseCode(result.m_responseCode)
{}

AmazonWebServiceResult(AmazonWebServiceResult&& result) :
m_payload(std::move(result.m_payload)),
m_responseHeaders(std::move(result.m_responseHeaders)),
m_responseCode(result.m_responseCode)
{}


inline const PAYLOAD_TYPE& GetPayload() const { return m_payload; }

inline PAYLOAD_TYPE TakeOwnershipOfPayload() { return std::move(m_payload); }

inline const Http::HeaderValueCollection& GetHeaderValueCollection() const { return m_responseHeaders; }

inline Http::HttpResponseCode GetResponseCode() const { return m_responseCode; }

private:
PAYLOAD_TYPE m_payload;
Http::HeaderValueCollection m_responseHeaders;
Http::HttpResponseCode m_responseCode;
};


}
