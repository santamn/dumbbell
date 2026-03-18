fn main() {
    #[cfg(feature = "gpu")]
    {
        println!("cargo:rerun-if-changed=src/kernel.cu");

        cc::Build::new()
            .cuda(true)
            .flag("-arch=sm_80") // Compute Capability 8.0 for A100
            .file("src/kernel.cu")
            .compile("kernel");

        println!("cargo:rustc-link-lib=curand");
    }
}
