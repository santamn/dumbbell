use std::env;
use std::fs::write;
use std::path::Path;

fn main() {
    let time: f64 = 10.0; // 総シミュレーション時間
    let delta_t: f64 = 2e-7; // 時間刻み幅
    let noise_scale: f64 = delta_t.sqrt(); // ノイズのスケール
    let steps = (time / delta_t) as usize; // シミュレーションステップ数
    let k: f64 = 1.5e6; // バネ定数
    let ensemble_size: u32 = 30_000; // アンサンブルサイズ
    let block_size: u32 = 256; // CUDAのブロックあたりのスレッド数

    let out_dir = env::var_os("OUT_DIR").unwrap();

    let constants_h = format!(
        "#pragma once\n\
         #define DELTA_T {delta_t:.15e}\n\
         #define K {k:.15e}\n\
         #define TIME {time:.15e}\n\
         #define STEPS {steps}\n\
         #define NOISE_SCALE {noise_scale:.15e}\n\
         #define ENSEMBLE_SIZE {ensemble_size}\n\
         #define THREADS_PER_BLOCK {block_size}\n"
    );
    write(Path::new(&out_dir).join("constants.h"), constants_h).unwrap();

    let constants_rs = format!(
        "pub const DELTA_T: f64 = {delta_t:.15e};\n\
         pub const K: f64 = {k:.15e};\n\
         pub const TIME: f64 = {time:.15e};\n\
         pub const STEPS: usize = {steps};\n\
         pub const NOISE_SCALE: f64 = {noise_scale:.15e};\n\
         pub const ENSEMBLE_SIZE: u32 = {ensemble_size};\n\
         pub const BLOCK_SIZE: u32 = {block_size};\n"
    );
    write(Path::new(&out_dir).join("constants.rs"), constants_rs).unwrap();

    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "gpu")]
    {
        println!("cargo:rerun-if-changed=src/simulation.cu");

        let status = std::process::Command::new("nvcc")
            .arg("-cubin")
            .arg("-arch=sm_80")
            .arg("-O3")
            .arg("-I")
            .arg(&out_dir)
            .arg("src/simulation.cu")
            .arg("-o")
            .arg(Path::new(&out_dir).join("simulation.cubin"))
            .status()
            .expect("Failed to run nvcc to build CUBIN");

        assert!(
            status.success(),
            "nvcc failed to compile simulation.cu to CUBIN"
        );
    }
}
