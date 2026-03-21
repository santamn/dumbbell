## Plan: CUDA計算の完全なTokio非同期化

現状の `rayon` 実装では計算自体はGPUにオフロードされているものの、`cudaDeviceSynchronize()` や同等のブロッキング関数によりCPUのOSスレッドが3つ待機状態（ブロック）となっています。このスレッド占有を解放し、Tokioを利用して真の非同期待機（グリーンスレッド / Taskのみの待機）に移行するための計画です。

**Steps**
1. **CUDA C部分の非同期API化** (`src/simulation.cu`)
   - `cudaDeviceSynchronize()` などのブロッキングAPIを削除。
   - すべてのGPU処理を `cudaStream_t` を用いて非同期化。
   - ホストとデバイス間のメモリ転送には `cudaMemcpyAsync` を使用し、ホスト側のメモリはページロックメモリ (`cudaMallocHost` など) として確保。
   - 計算の完了検知のため、`cudaEvent_t` を作成・記録 (`cudaEventRecord`) し、Rust側にイベント等のハンドル（ポインタ等）を返す初期化関数を作成。
   - 状態確認用として、`cudaEventQuery` を呼び出して完了状態（`cudaSuccess` または `cudaErrorNotReady`）を返すポーリング用のC関数を追加。
   - メモリ解放およびリソース破棄を行うC関数を追加。

2. **Rust側でのFuture実装** (`src/simulation.rs` または `src/statistics.rs`)
   - 上記のC関数群を `extern "C"` でバインディング。
   - `Future` トレイトを実装するラッパー構造体（例: `GpuTaskFuture`）を作成。
   - `Future::poll` メソッド内で、C側の状態確認用関数 (`cudaEventQuery` 相当) を呼び出す。
     - 完了していれば計算結果を取り出し `Poll::Ready` を返す。
     - 未完了の場合は、Wakerを保存し、一旦 `Poll::Pending` を返す。この際、ビジーループを避けるため、Tokioの機能（別スレッドからの起床や `tokio::time::sleep` など）と連携して再度 `poll` が発火するよう工夫する。

3. **`statistics.rs` のTokio非同期化** (`src/statistics.rs`)
   - GPUの並列処理（3 GPU分）をRayonの `into_par_iter` から、Tokioの `tokio::spawn` または `futures::future::join_all` を使った非同期タスク群の立ち上げへ変更。
   - 各GPUタスクで `GpuTaskFuture` を `.await` し、CPU（OSスレッド）をブロックせずに全GPUの計算終了を待機。
   - すべての結果が揃った後、平均・有効拡散などの統計値を合算して返す。

**Relevant files**

- `src/simulation.cu` — `calculate_displacement_sum_on_gpu`の非同期化（内部でストリームやイベントを使用する形への解体）、非同期状態を問い合わせる API (`cudaEventQuery` 利用) を新設。
- `src/simulation.rs` / `src/statistics.rs` — FFI定義の更新、状態確認を行う `Future` トレイト実装の追加。
- `src/statistics.rs` — 統計計算のメインループ (`statistics` 関数) を `async fn` に変更し、Rayonから `tokio::join!` または `join_all` へ移行。
- `Cargo.toml` — `tokio` （非同期ランタイム）および必要ならば `futures` クレートの追加。

**Verification**

1. `cargo add tokio -F full` コマンド等によりTokioを依存関係に追加。
2. 変更後、コンパイルエラー（非同期関数の呼び出しコンテキストなど）を解消。
3. 実行時のGPU使用率と、CPU（Topコマンド等）でのスレッド数およびアイドル状態を確認し、意図通りスレッドを浪費せずに並列実行できているか確認（実際は3スレッドなのでTopから視認しにくい場合はデバッガ等でOSスレッドの数を比較する）。

**Decisions**

- FFIを通じたC側からのコールバック呼び出し（`cudaStreamAddCallback` -> RustのWaker）は、スレッド間競合やライフタイム管理が複雑になりバグの温床となるため、Rust側からポーリング（`cudaEventQuery` をチェックする方式）もしくはバックグラウンドの監視スレッドによる非同期処理を採用する前提とする。

**Further Considerations**

1. Rust側の非同期処理において、単純に定期的にスリープしながら `cudaEventQuery` を叩く（ポーリング方式）か、CUDAの完了を待機するための軽量な専用OSスレッド（CUDA API自身でブロックさせる役目）を1つだけ立てて一括監視する方式か、どちらが実装として望ましいか？（前者のタイマー・ポーリング方式の方が複雑なFFIのライフタイムを避けられ安全かつTokioフレンドリーです）。