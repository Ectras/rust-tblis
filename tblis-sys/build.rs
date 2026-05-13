use std::{env, path::PathBuf};

use cmake::Config;

fn main() {
    // Build tblis
    let extern_path = PathBuf::from("extern");
    let tblis_path = extern_path.join("tblis");
    unsafe {
        // Workaround until https://github.com/flame/blis/issues/741 is fixed
        env::set_var("CC", "/usr/bin/gcc");
        env::set_var("CXX", "/usr/bin/g++");
    }
    let dst = Config::new(&tblis_path)
        .configure_arg("-DENABLE_SHARED=OFF")
        .configure_arg("-DLABEL_TYPE=size_t")
        .build();

    // Link it
    println!("cargo:rustc-link-search={}", dst.join("lib").display());
    println!(
        "cargo:rustc-link-search={}",
        dst.join("lib").join("tblis").display()
    );
    println!("cargo:rustc-link-lib=static=tblis");
    println!("cargo:rustc-link-lib=static=tci");
    println!("cargo:rustc-link-lib=static=blis_core");
    println!("cargo:rustc-link-lib=static=blis_tblis");
    println!("cargo:rustc-link-lib=hwloc");
    println!("cargo:rustc-link-lib=atomic");

    // Link the C++ standard library
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    }

    // Generate bindings
    let header = dst.join("include").join("tblis.h");
    let bindings = bindgen::Builder::default()
        .clang_arg(format!(
            "--include-directory={}",
            dst.join("include").display()
        ))
        .header(header.to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Unable to write bindings");
}
