use std::env;
use std::path::Path;

/// GPU機能が有効な場合に、CUDAカーネル(simulation.cu)をnvccでCUBINにコンパイルする。
/// シミュレーションの定数は実行時にTOML設定から読み込むため、ここでは何も生成しない。
fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // ビルドスクリプトでは cfg!(feature) は使えないため、Cargoが設定する環境変数で判定する
    if env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }

    println!("cargo:rerun-if-changed=src/simulation.cu");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let status = std::process::Command::new("nvcc")
        .arg("-cubin")
        .arg("-arch=sm_80") // NVIDIA A100 (Ampere)
        .arg("-O3")
        .arg("src/simulation.cu")
        .arg("-o")
        .arg(Path::new(&out_dir).join("simulation.cubin"))
        .status()
        .expect("nvccの実行に失敗(CUDA Toolkitがインストールされているか確認してください)");

    assert!(
        status.success(),
        "nvccによるsimulation.cuのコンパイルに失敗"
    );
}
