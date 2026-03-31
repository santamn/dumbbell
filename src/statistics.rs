pub struct Statistics {
    pub effective_diffusion: f64,
    pub first_passage_time: f64,
    pub nonlinear_mobility: f64,
}

pub use backend::{BulkBuffer, statistics};

#[cfg(feature = "gpu")]
mod backend {
    use super::{Statistics, diffusion, nonlinear_mobility};
    use crate::simulation::{ENSEMBLE_SIZE, TIME};
    use std::ffi::c_void;
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::sync::OnceLock;
    use std::sync::mpsc;
    use std::thread;
    use tokio::sync::oneshot;

    unsafe extern "C" {
        /// Pinned Memoryを確保するための関数
        unsafe fn alloc_pinned_f64_memories(n: usize) -> *mut f64;

        /// Pinned Memoryを解放するための関数
        unsafe fn free_pinned_f64_memories(ptr: *mut f64);

        /// Device Memoryを確保するための関数
        unsafe fn alloc_device_f64_memories(n: usize, device_id: u64) -> *mut f64;

        /// Device Memoryを解放するための関数
        unsafe fn free_device_f64_memories(ptr: *mut f64, device_id: u64);

        /// GPUの全ての作業が終わるまでCPUをブロックして待機する関数
        unsafe fn synchronize_gpu_device(device_id: u64);

        /// GPUを用いてシミュレーション結果の総和の計算を非同期で行う関数
        unsafe fn async_calculate_displacements_sum_on_gpu(
            rust_callback: unsafe extern "C" fn(*mut c_void, f64, f64), // 計算結果を送るためのコールバック
            sender: *mut c_void, // 計算結果を送るためのチャネル （oneshot::Senderのポインタ）
            host_disp_ptr: *mut f64, // 確保済みのPinned Memory
            host_sq_disp_ptr: *mut f64, // 同上
            dev_disp_ptr: *mut f64, // 事前確保済みのデバイスメモリ
            dev_sq_disp_ptr: *mut f64, // 同上
            device_id: u64,
            seed: u64,
            length: f64,
            force_x: f64,
        );
    }

    /// GPUと非同期通信するための一括確保されたページロックメモリとデバイスメモリを管理する構造体
    pub struct BulkBuffer {
        device_id: u64,
        disp_array: *mut f64,
        sq_disp_array: *mut f64,
        dev_disp_array: *mut f64,
        dev_sq_disp_array: *mut f64,
        capacity: usize,
        is_freed: bool,
    }

    // SAFETY: BulkBuffer はCUDAページロックメモリとデバイスメモリを管理するラッパーであり、
    //         Drop時に適切に解放されるため、Box と同様に Send の対象となる
    unsafe impl Send for BulkBuffer {}

    impl BulkBuffer {
        pub fn new(device_id: u64, total_tasks: usize) -> Self {
            unsafe {
                Self {
                    device_id,
                    disp_array: alloc_pinned_f64_memories(total_tasks),
                    sq_disp_array: alloc_pinned_f64_memories(total_tasks),
                    dev_disp_array: alloc_device_f64_memories(total_tasks, device_id),
                    dev_sq_disp_array: alloc_device_f64_memories(total_tasks, device_id),
                    capacity: total_tasks,
                    is_freed: false,
                }
            }
        }

        /// 指定したインデックスの書き込み先ポインタを取得する
        pub fn get_pointers(&self, index: usize) -> Pointers {
            assert!(index < self.capacity); // 安全のため、インデックスが容量内に収まっていることを確認する

            unsafe {
                Pointers {
                    // .add(index) は自動的に sizeof(f64)*index 分だけアドレスを加算する
                    disp: self.disp_array.add(index),
                    sq_disp: self.sq_disp_array.add(index),
                    dev_disp: self.dev_disp_array.add(index),
                    dev_sq_disp: self.dev_sq_disp_array.add(index),
                }
            }
        }

        /// 正常完了時に呼び出し、非同期で安全にクリーンアップを行うメソッド
        pub async fn dispose(&mut self) {
            let device_id = self.device_id;
            // 生ポインタはSendを実装していないため、usizeにキャストしてクロージャに渡す
            let disp_addr = self.disp_array as usize;
            let sq_disp_addr = self.sq_disp_array as usize;
            let dev_disp_addr = self.dev_disp_array as usize;
            let dev_sq_disp_addr = self.dev_sq_disp_array as usize;

            // メモリを解放する前に、GPUがこのメモリへの非同期書き込みをすべて完了するまで待つ必要があるが、
            // 非同期ランタイムのワーカースレッドをブロックしないために spawn_blocking に逃がす
            tokio::task::spawn_blocking(move || {
                // ここは通常のOSスレッドなので、GPUの完了を待つためにブロックしても問題ない
                unsafe {
                    synchronize_gpu_device(device_id);
                    free_pinned_f64_memories(disp_addr as *mut f64);
                    free_pinned_f64_memories(sq_disp_addr as *mut f64);
                    free_device_f64_memories(dev_disp_addr as *mut f64, device_id);
                    free_device_f64_memories(dev_sq_disp_addr as *mut f64, device_id);
                }
            })
            .await
            .unwrap();

            self.is_freed = true;
        }
    }

    impl Drop for BulkBuffer {
        fn drop(&mut self) {
            if !self.is_freed {
                // Drop時にまだ解放されていない場合は、ブロックしてでもリソースを解放する
                // これは安全性を優先するための最後の手段であり、通常は dispose() を呼び出すべき
                unsafe {
                    synchronize_gpu_device(self.device_id);
                    free_pinned_f64_memories(self.disp_array);
                    free_pinned_f64_memories(self.sq_disp_array);
                    free_device_f64_memories(self.dev_disp_array, self.device_id);
                    free_device_f64_memories(self.dev_sq_disp_array, self.device_id);
                }
            }
            self.is_freed = true;
        }
    }

    /// 各シミュレーションに渡すためのポインタのセット
    #[derive(Clone, Copy)]
    pub struct Pointers {
        pub disp: *mut f64,
        pub sq_disp: *mut f64,
        pub dev_disp: *mut f64,
        pub dev_sq_disp: *mut f64,
    }

    // SAFETY:
    //  1. スレッド非依存: Pointers はスレッドローカルストレージではなく、OS/CUDAによって割り当てられたグローバルな領域であるため、
    //                 別スレッドへ移動しても有効
    //  2. 排他性: 各タスクにはBulkBuffer内の重複しない一意のインデックスを指すポインタが割り当てられるため、
    //            別々のスレッドにSendされて各々が独立して書き込みを行っても、エイリアシングによる未定義動作は発生しない
    //  3. 寿命: 参照先のメモリは親となるBulkBufferが破棄されるまで（すべてのGPUタスクが終わるまで）有効である
    unsafe impl Send for Pointers {}

    // CUDAドライバからの通知をOSスレッドで受け取るためのメッセージ
    struct CallbackMessage {
        sender_ptr: *mut c_void,
        disp_sum: f64,
        sq_disp_sum: f64,
    }

    // 生ポインタを含んでいるが専用スレッドに渡して使用するためSendを実装
    // SAFETY: CallbackMessage はGPUからのコールバックで受け取る生ポインタを含むが、
    //         これらのポインタは専用のOSスレッド内でのみ使用され、他のスレッドに送られることはないため Send を実装しても安全
    unsafe impl Send for CallbackMessage {}

    static CALLBACK_TX: OnceLock<mpsc::Sender<CallbackMessage>> = OnceLock::new();

    fn get_callback_tx() -> mpsc::Sender<CallbackMessage> {
        CALLBACK_TX
            .get_or_init(|| {
                let (tx, rx) = mpsc::channel::<CallbackMessage>();
                // アプリの起動中ずっと動き続けてTokioのスレッドをWakeする専用のOSスレッドを立ち上げる
                thread::spawn(move || {
                    for msg in rx {
                        // ここは通常のOSスレッドなので、TokioのWakerを起動させる処理の重さを気にせずsendできる
                        let sender = unsafe {
                            Box::from_raw(msg.sender_ptr as *mut oneshot::Sender<(f64, f64)>)
                        };
                        let _ = sender.send((msg.disp_sum, msg.sq_disp_sum));
                    }
                });
                tx
            })
            .clone()
    }

    unsafe extern "C" fn gpu_done_callback(sender: *mut c_void, disp_sum: f64, sq_disp_sum: f64) {
        // CUDAドライバ管轄のOSスレッドをブロックしないよう、専用スレッドへメッセージとして送る
        let tx = get_callback_tx();
        let _ = tx.send(CallbackMessage {
            sender_ptr: sender,
            disp_sum,
            sq_disp_sum,
        });
    }

    /// GPUを用いてアンサンブル平均を非同期で計算する関数
    ///
    /// この関数は呼び出されると直ちにGPUに計算を投げ、完了まで現在のTokioタスクをOSスレッドをブロックすることなく完全にスリープさせる
    pub async fn statistics(device_id: u64, length: f64, force: f64, ptrs: Pointers) -> Statistics {
        // 長さと外力からハッシュ値を生成して、GPU側の乱数生成器のシードとして利用する
        let mut hasher = DefaultHasher::new();
        length.to_bits().hash(&mut hasher);
        force.to_bits().hash(&mut hasher);

        // 結果を受け取るための1回限りの通信チャネル(oneshot)を作成
        let (tx, rx) = oneshot::channel::<(f64, f64)>();

        unsafe {
            // CUDAの非同期関数を呼び出す
            // GPUへのコマンド送信をスケジュールするだけで関数自体は即座にリターンされるので、ブロックされない
            async_calculate_displacements_sum_on_gpu(
                gpu_done_callback,
                // Cのコールバックに持たせるためにSenderをヒープに置き、所有権を放棄して生ポインタに変換する
                Box::into_raw(Box::new(tx)) as *mut c_void,
                ptrs.disp,
                ptrs.sq_disp,
                ptrs.dev_disp,
                ptrs.dev_sq_disp,
                device_id,
                hasher.finish(),
                length,
                force,
            );
        }

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
