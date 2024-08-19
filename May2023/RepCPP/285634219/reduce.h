template<typename T>
inline T warpReduceSum(T val, nd_item<2> &item)
{
auto sg = item.get_sub_group();
#pragma unroll
for (int mask = SG/2; mask > 0; mask >>= 1)
val += permute_group_by_xor(sg, val, mask);
return val;
}


template<typename T>
inline T blockReduceSum(T val, nd_item<2> &item, T *shared)
{
#ifdef WAVE64
int lane = item.get_local_id(1) & 0x3f; 
int wid = item.get_local_id(1) >> 6;    
#else
int lane = item.get_local_id(1) & 0x1f; 
int wid = item.get_local_id(1) >> 5;    
#endif

val = warpReduceSum<T>(val, item);

if (lane == 0)
shared[wid] = val;

item.barrier(access::fence_space::local_space);

val = (item.get_local_id(1) < (item.get_local_range(1) / (float)SG))
? shared[lane] : (T)(0.0f);
val = warpReduceSum<T>(val, item);

return val;
}

template<typename T>
inline T warpReduceMax(T val, nd_item<2> &item)
{
auto sg = item.get_sub_group();
#pragma unroll
for (int mask = SG/2; mask > 0; mask >>= 1)
val = sycl::max(val, permute_group_by_xor(sg, val, mask));
return val;
}


template<typename T>
inline T blockReduceMax(T val, nd_item<2> &item, T *shared)
{

#ifdef WAVE64
int lane = item.get_local_id(1) & 0x3f; 
int wid = item.get_local_id(1) >> 6;    
#else
int lane = item.get_local_id(1) & 0x1f; 
int wid = item.get_local_id(1) >> 5;    
#endif

val = warpReduceMax(val, item); 

if (lane == 0)  
shared[wid] = val;

item.barrier(access::fence_space::local_space);


val = (item.get_local_id(1) < (item.get_local_range(1) / (float)SG))
? shared[lane]
: -1e20f;
val = warpReduceMax(val, item);

return val;
}
