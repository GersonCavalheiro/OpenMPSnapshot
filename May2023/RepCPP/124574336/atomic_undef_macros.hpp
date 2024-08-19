

#if defined(BOOST_INTEL)

#pragma push_macro("atomic_compare_exchange")
#undef atomic_compare_exchange

#pragma push_macro("atomic_compare_exchange_explicit")
#undef atomic_compare_exchange_explicit

#pragma push_macro("atomic_exchange")
#undef atomic_exchange

#pragma push_macro("atomic_exchange_explicit")
#undef atomic_exchange_explicit

#pragma push_macro("atomic_is_lock_free")
#undef atomic_is_lock_free

#pragma push_macro("atomic_load")
#undef atomic_load

#pragma push_macro("atomic_load_explicit")
#undef atomic_load_explicit

#pragma push_macro("atomic_store")
#undef atomic_store

#pragma push_macro("atomic_store_explicit")
#undef atomic_store_explicit


#endif 


