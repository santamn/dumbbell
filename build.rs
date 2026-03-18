fn main() {
    #[cfg(feature = "gpu")]
    {
        println!("cargo:rerun-if-changed=src/simulation.cu");

        cc::Build::new()
            .cuda(true)
            .flag("-arch=sm_80") // Compute Capability 8.0 for A100
            .file("src/simulation.cu")
            .compile("simulation");

        println!("cargo:rustc-link-lib=curand");
    }
}
