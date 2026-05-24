#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

// openmp_sys takes care of linking OpenMP which is used by tblis.
extern crate openmp_sys;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
