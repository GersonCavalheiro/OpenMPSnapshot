

#if defined(BOOST_INTEL)

#pragma pop_macro("atomic_compare_exchange")
#pragma pop_macro("atomic_compare_exchange_explicit")
#pragma pop_macro("atomic_exchange")
#pragma pop_macro("atomic_exchange_explicit")
#pragma pop_macro("atomic_is_lock_free")
#pragma pop_macro("atomic_load")
#pragma pop_macro("atomic_load_explicit")
#pragma pop_macro("atomic_store")
#pragma pop_macro("atomic_store_explicit")

#endif 
