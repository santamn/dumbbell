use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let delta_t: f64 = 2e-7; // 時間刻み幅
    let k: f64 = 1.5e6; // バネ定数
    let time: f64 = 10.0; // 総シミュレーション時間
    let steps: usize = (time / delta_t) as usize; // シミュレーションステップ数
    let noise_scale: f64 = delta_t.sqrt(); // ノイズのスケール
    let ensemble_size: u64 = 30_000; // アンサンブルサイズ

    let out_dir = env::var_os("OUT_DIR").unwrap();

    let mut constants_h = File::create(Path::new(&out_dir).join("constants.h")).unwrap();
    writeln!(constants_h, "#pragma once").unwrap();
    writeln!(constants_h, "#define DELTA_T {:.15e}", delta_t).unwrap();
    writeln!(constants_h, "#define K {:.15e}", k).unwrap();
    writeln!(constants_h, "#define TIME {:.15e}", time).unwrap();
    writeln!(constants_h, "#define STEPS {}", steps).unwrap();
    writeln!(constants_h, "#define NOISE_SCALE {:.15e}", noise_scale).unwrap();
    writeln!(constants_h, "#define ENSEMBLE_SIZE {}", ensemble_size).unwrap();

    let mut constants_rs = File::create(Path::new(&out_dir).join("constants.rs")).unwrap();
    writeln!(constants_rs, "pub const DELTA_T: f64 = {:.15e};", delta_t).unwrap();
    writeln!(constants_rs, "pub const K: f64 = {:.15e};", k).unwrap();
    writeln!(constants_rs, "pub const TIME: f64 = {:.15e};", time).unwrap();
    writeln!(constants_rs, "pub const STEPS: usize = {};", steps).unwrap();
    writeln!(
        constants_rs,
        "pub const NOISE_SCALE: f64 = {:.15e};",
        noise_scale
    )
    .unwrap();
    writeln!(
        constants_rs,
        "pub const ENSEMBLE_SIZE: u64 = {};",
        ensemble_size
    )
    .unwrap();

    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "gpu")]
    {
        println!("cargo:rerun-if-changed=src/simulation.cu");

        cc::Build::new()
            .cuda(true)
            .flag("-arch=sm_80")
            .include(&out_dir)
            .file("src/simulation.cu")
            .compile("simulation");

        println!("cargo:rustc-link-lib=curand");
    }
}
