pub struct Statistics {
    pub effective_diffusion: f64,
    pub first_passage_time: f64,
    pub nonlinear_mobility: f64,
}

pub use backend::{BulkPinnedBuffer, statistics};

#[cfg(feature = "gpu")]
mod backend {
    use super::{Statistics, diffusion, nonlinear_mobility};
    use crate::simulation::{ENSEMBLE_SIZE, TIME};
    use std::ffi::c_void;
    use tokio::sync::oneshot;

    unsafe extern "C" {
        /// Pinned Memoryを確保するための関数
        unsafe fn alloc_pinned_f64_memories(n: usize) -> *mut f64;

        /// Pinned Memoryを解放するための関数
        unsafe fn free_pinned_f64_memories(ptr: *mut f64);

        /// GPUの全ての作業が終わるまでCPUをブロックして待機する関数
        unsafe fn synchronize_gpu_device(device_id: u64);

        /// 特定のストリームの完了を同期的に待機する関数
        unsafe fn synchronize_cuda_stream(stream: *mut c_void);

        /// GPUのストリームを破棄する関数
        unsafe fn destroy_cuda_stream(stream: *mut c_void);

        /// GPUを用いてシミュレーション結果の総和の計算を非同期で行う関数
        ///
        /// 返り値はGPUへのコマンド送信をスケジュールした後に即座に返されるCUDAストリームのポインタであり、呼び出し元で破棄する必要がある
        unsafe fn async_calculate_displacements_sum_on_gpu(
            rust_callback: unsafe extern "C" fn(*mut c_void, f64, f64),
            sender: *mut c_void,
            host_disp_ptr: *mut f64,    // 確保済みのPinned Memory
            host_sq_disp_ptr: *mut f64, // 同上
            device_id: u64,
            seed: u64,
            length: f64,
            force_x: f64,
        ) -> *mut c_void;
    }

    /// GPUと非同期通信するための一括確保されたPinned Memory（ページロックメモリ）を管理する構造体
    pub struct BulkPinnedBuffer {
        device_id: u64,
        disp_array: *mut f64,
        sq_disp_array: *mut f64,
        capacity: usize,
    }

    // SAFETY: BulkPinnedBuffer は実質的にヒープ上に確保された CUDA ページロックメモリを所有する独自のラッパーであり、
    //         Drop時に適切に解放されるため、Box と同様に Send と Sync の対象となる。
    unsafe impl Send for BulkPinnedBuffer {}
    unsafe impl Sync for BulkPinnedBuffer {}

    impl BulkPinnedBuffer {
        pub fn new(device_id: u64, total_tasks: usize) -> Self {
            unsafe {
                Self {
                    device_id,
                    disp_array: alloc_pinned_f64_memories(total_tasks),
                    sq_disp_array: alloc_pinned_f64_memories(total_tasks),
                    capacity: total_tasks,
                }
            }
        }

        /// 指定したインデックスの書き込み先ポインタを取得する
        pub fn get_pointers(&self, index: usize) -> Pointers {
            assert!(index < self.capacity); // 安全のため、インデックスが容量内に収まっていることを確認する

            unsafe {
                Pointers {
                    // .add(index) は自動的に sizeof(f64) 分だけアドレスを計算する
                    disp: self.disp_array.add(index),
                    sq_disp: self.sq_disp_array.add(index),
                }
            }
        }
    }

    impl Drop for BulkPinnedBuffer {
        fn drop(&mut self) {
            unsafe {
                // メモリを解放する前に、GPUがこのメモリへの非同期書き込みをすべて完了するまで強制的にブロックする
                synchronize_gpu_device(self.device_id);

                // GPUが完全に止まったことが保証されたので、安全に解放する
                free_pinned_f64_memories(self.disp_array);
                free_pinned_f64_memories(self.sq_disp_array);
            }
        }
    }

    /// 各シミュレーションに渡すためのポインタのペア
    #[derive(Clone, Copy)]
    pub struct Pointers {
        pub disp: *mut f64,
        pub sq_disp: *mut f64,
    }

    // SAFETY: Pointers は BulkPinnedBuffer 内の互いに重複しない独立したインデックスの生ポインタペアであり、
    //         各々のタスク（スレッド）で別々に扱われ GPU に送信するため、Send と Sync を実装しても安全である。
    unsafe impl Send for Pointers {}
    unsafe impl Sync for Pointers {}

    /// CUDAストリーム(ポインタ)をラップし、SendとDropを実装するための構造体
    struct AsyncCudaStream(*mut c_void);

    // SAFETY: CUDAストリームはスレッド間で共有・送信しても安全なハンドルであり、
    //         Drop実装で適切にクリーンアップされるため、Sendを実装しても安全
    unsafe impl Send for AsyncCudaStream {}

    impl Drop for AsyncCudaStream {
        fn drop(&mut self) {
            unsafe {
                // Tokioタスクがキャンセル（Drop）された場合でも、該当ストリームのGPU処理が完全に終わるまで同期的（OSスレッド）に待機する
                // キャンセル後にPointersが再利用された際に古いストリームが遅れてメモリを上書きする危険や、
                // C側に渡した生ポインタ (sender) を使ったコールバックが想定外のタイミングで発火する事態を未然に防ぐ
                synchronize_cuda_stream(self.0);
                destroy_cuda_stream(self.0);
            }
        }
    }

    unsafe extern "C" fn gpu_done_callback(sender: *mut c_void, disp_sum: f64, sq_disp_sum: f64) {
        // user_data として渡された生ポインタから Box<Sender> を復元し、リソースの所有権を取り戻す
        let sender = unsafe { Box::from_raw(sender as *mut oneshot::Sender<(f64, f64)>) };
        // async 側の rx.await で待機しているタスクへ計算結果を送信して起床させる
        let _ = sender.send((disp_sum, sq_disp_sum));
    }

    /// GPUを用いてアンサンブル平均を非同期で計算する関数
    ///
    /// この関数は呼び出されると直ちにGPUに計算を投げ、完了まで現在のTokioタスクをOSスレッドをブロックすることなく完全にスリープさせる
    pub async fn statistics(device_id: u64, length: f64, force: f64, ptrs: Pointers) -> Statistics {
        // 結果を受け取るための1回限りの通信チャネル(oneshot)を作成
        let (tx, rx) = oneshot::channel::<(f64, f64)>();

        let _stream = AsyncCudaStream(unsafe {
            // CUDAの非同期関数を呼び出す
            // GPUへのコマンド送信をスケジュールするだけで関数自体は即座にリターンされるので、ブロックされない
            async_calculate_displacements_sum_on_gpu(
                gpu_done_callback,
                Box::into_raw(Box::new(tx)) as *mut c_void, // Cのコールバックに持たせるためにSenderをヒープに置き、所有権を放棄して生ポインタに変換する
                ptrs.disp,
                ptrs.sq_disp,
                device_id,
                1,
                length,
                force,
            )
        });

        // CUDAの計算が終わって gpu_done_callback が呼ばれるまで、Tokioタスクを非同期待機させる
        let (disp_sum, sq_disp_sum) = rx.await.expect("GPUでの計算完了のコールバックの受信に失敗");

        let mean_displacement = disp_sum / ENSEMBLE_SIZE as f64;
        let mean_square_displacement = sq_disp_sum / ENSEMBLE_SIZE as f64;
        let mean_speed = mean_displacement / TIME;

        Statistics {
            effective_diffusion: diffusion(mean_displacement, mean_square_displacement, TIME),
            first_passage_time: 1.0 / mean_speed.abs(),
            nonlinear_mobility: nonlinear_mobility(mean_speed, force),
        }
    }
}

#[cfg(not(feature = "gpu"))]
mod backend {
    use super::{Statistics, diffusion, nonlinear_mobility};
    use crate::simulation::{ENSEMBLE_SIZE, Particle, STEPS, TIME};
    use nalgebra::Vector2;
    use rand::{SeedableRng, rngs::SmallRng};
    use rayon::prelude::*;

    /// アンサンブル平均を用いて、非線形移動度、整流尺度、有効拡散係数を計算する
    pub fn statistics(length: f64, force: f64) -> Statistics {
        let force_vec = Vector2::new(force, 0.0);
        let (mean_displacement, mean_square_displacement) = (0..ENSEMBLE_SIZE)
            .into_par_iter()
            .map(|i| {
                let rng = SmallRng::seed_from_u64(i);
                let mut particle = Particle::new(rng, length, force_vec);
                let start = particle.now().position.x;
                let delta_x = particle.nth(STEPS).unwrap().position.x - start; // 移動距離

                (delta_x, delta_x * delta_x) // 変位, 二乗変位
            })
            .reduce_with(|(a, aa), (x, xx)| (a + x, aa + xx))
            .map(|(sum, sq_sum)| (sum / ENSEMBLE_SIZE as f64, sq_sum / ENSEMBLE_SIZE as f64))
            .unwrap();

        let mean_speed = mean_displacement / TIME;

        Statistics {
            effective_diffusion: diffusion(mean_displacement, mean_square_displacement, TIME),
            first_passage_time: 1.0 / mean_speed,
            nonlinear_mobility: nonlinear_mobility(mean_speed, force),
        }
    }

    pub async fn statistics_async(_device_id: u64, length: f64, force: f64) -> Statistics {
        // CPU fallback for async. Just spawn blocking.
        tokio::task::spawn_blocking(move || statistics(length, force))
            .await
            .unwrap()
    }
}

#[allow(dead_code)]
pub fn particle_distribution() {
    todo!("アンサンブル平均により、粒子の位置分布を推定する機能を実装する予定")
}

/// 非線形移動度 μ(f) = ⟨v⟩/|f|
fn nonlinear_mobility(mean_speed: f64, force: f64) -> f64 {
    mean_speed / force
}

/// 有効拡散係数 D_eff = (⟨x²⟩ - ⟨x⟩²)/2t
fn diffusion(mean_displacement: f64, mean_square_displacement: f64, time: f64) -> f64 {
    (mean_square_displacement - mean_displacement * mean_displacement) / (2.0 * time)
}

/// 整流尺度 α = |μ(f) - μ(-f)| / (μ(f) + μ(-f))å
#[allow(dead_code)]
pub fn alpha(mu_forward: f64, mu_backward: f64) -> f64 {
    (mu_forward - mu_backward).abs() / (mu_forward + mu_backward)
}
