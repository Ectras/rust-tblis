#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

extern crate openmp_sys;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
