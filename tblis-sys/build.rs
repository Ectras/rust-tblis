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
    let mut config = Config::new(&tblis_path);
    config.configure_arg("-DENABLE_SHARED=false");
    config.configure_arg("-DLABEL_TYPE=size_t");

    // Optionally enable hwloc support
    let use_hwloc = cfg!(feature = "hwloc");
    config.configure_arg(format!("-DENABLE_HWLOC={}", use_hwloc));

    // Build with cmake
    let dst = config.build();

    // Use pkg-config to find the built library
    unsafe {
        std::env::set_var("PKG_CONFIG_PATH", dst.join("lib").join("pkgconfig"));
    }

    // Extract linker flags using pkg-config
    pkg_config::Config::new()
        .statik(true)
        .env_metadata(false)
        .probe("tblis")
        .unwrap();

    // Generate bindings
    let header = dst.join("include").join("tblis.h");
    let bindings = bindgen::Builder::default()
        .clang_arg(format!(
            "--include-directory={}",
            dst.join("include").display()
        ))
        .header(header.to_str().unwrap())
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Unable to write bindings");
}
