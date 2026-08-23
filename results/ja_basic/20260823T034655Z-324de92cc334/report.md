# モデル比較レポート: ja_basic

日本語の基本タスク 8 問で生成速度・VRAM・再現条件を比較する

## 再現条件

| 項目 | 値 |
|---|---|
| run_fingerprint | 324de92cc334366bd7cfca41816ce81082977bf964051a9e75c2e77d75133e24 |
| run_id | aa1ae767c9fa |
| 実行開始 (UTC) | 2026-08-23T03:46:55.303138Z |
| スイート | suites/ja_basic.toml |
| suite_sha256 | 18ad5d4f3feafbe7071ab6adac32fbe2b5c9f1486692743925892fc16ec1a154 (ベースファイルのハッシュ。実効入力は reproduction を参照) |
| 絞り込み (--models/--limit) | (絞り込みなし: 全モデル・全プロンプト) |
| 設定 | configs/default.toml |
| config_sha256 | f9b82bcc204b7e23b08bd39078f5e318204d010a9ae066a34b1b4c90e313dc10 |
| ウォームアップ回数 | 1 |
| プロンプト数 | 8 |
| モデル数 | 3 |
| レコードスキーマ版 | 1 |

実行マニフェスト: `ja_basic` の各モデル 1 本ずつ (同ディレクトリの `manifests/`)。

## モデル比較

| model_id | served_name | quantization | 設定コンテキスト長 | 生成速度 t/s (中央値) | 速度出典 | プロンプト処理速度 t/s (中央値) | 壁時計速度 t/s (中央値) | VRAM 見積り GiB | VRAM 実測増分 GiB | 測定数 n/N |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-14b | qwen3:14b-q4_K_M | Q4_K | 16384 | 61.7 | eval | 2347.1 | 61.1 | 11.98 | 11.13 | 8/8 |
| gpt-oss-20b | gpt-oss:20b | MXFP4 | 16384 | 133.8 | eval | 2256.2 | 130.8 | 12.51 | 12.51 | 8/8 |
| qwen3-8b | qwen3:8b-q4_K_M | Q4_K | 16384 | 105.8 | eval | 3743.0 | 104.2 | 7.26 | 7.27 | 8/8 |

欠測は `—`。集計は `phase=measure` かつ成功したレコードのみで、`測定数 n/N` は「測定できた件数 / 試行件数」。

## 応答

### summarize

タグ: 要約

system:

> あなたは日本語の技術文書を扱うアシスタントです。

プロンプト:

> 次の文章を3行で要約してください。
>
> 大規模言語モデルをローカルの GPU で動かす場合、モデルの重み・KV キャッシュ・
> 推論ランタイムの作業領域の3つが VRAM を消費する。重みは量子化の方式で決まり、
> KV キャッシュはコンテキスト長にほぼ比例して増える。したがって、同じモデルでも
> コンテキスト長を伸ばすと搭載メモリに収まらなくなることがある。収まらない分は
> CPU 側のメモリへ退避されるが、そのとき生成速度は大きく低下する。

#### qwen3-14b

- 生成速度: 61.9 t/s (出典: eval) / 完了トークン: 280 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=1)</summary>

> 大規模言語モデルをローカルGPUで実行する際、VRAMを消費する要素はモデルの重み、KVキャッシュ、推論ランタイムの作業領域の3つである。KVキャッシュはコンテキスト長に比例して増加し、長さを伸ばすとVRAMを超える可能性がある。その場合、CPUメモリに退避させることで生成速度が低下する。

</details>

#### gpt-oss-20b

- 生成速度: 132.7 t/s (出典: eval) / 完了トークン: 370 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=10)</summary>

> 大規模言語モデルをローカルGPUで動かすと、重み・KVキャッシュ・推論ランタイムの3要素がVRAMを消費します。  
> 重みは量子化方式で決まり、KVキャッシュはコンテキスト長に比例して増大するため、同一モデルでもコンテキスト長を伸ばすとGPUメモリを圧迫します。  
> メモリ不足時はCPUメモリへ退避され、生成速度が大幅に低下します。

</details>

#### qwen3-8b

- 生成速度: 105.9 t/s (出典: eval) / 完了トークン: 333 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=19)</summary>

> 大規模言語モデルのローカルGPU実行では、モデル重み、KVキャッシュ、推論ランタイムの作業領域がVRAMを消費する。  
> 重みは量子化方式で、KVキャッシュはコンテキスト長に比例して増加する。  
> コンテキスト長を長くするとGPUメモリ不足が発生し、CPUメモリへの退避により生成速度が大幅に低下する。

</details>

### paraphrase

タグ: 言い換え

プロンプト:

> 次の一文を、意味を変えずにより平易な日本語へ言い換えてください。
> 言い換えた文だけを出力してください。
>
> 「当該処理は、入力値の妥当性検証が完了した後に限り実行されるものとする。」

#### qwen3-14b

- 生成速度: 61.8 t/s (出典: eval) / 完了トークン: 512 / finish_reason: length

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=2)</summary>

> この処理は、入

</details>

#### gpt-oss-20b

- 生成速度: 133.7 t/s (出典: eval) / 完了トークン: 249 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=11)</summary>

> 入力値の妥当性をチェックしたあとだけ、処理を実行します。

</details>

#### qwen3-8b

- 生成速度: 105.9 t/s (出典: eval) / 完了トークン: 512 / finish_reason: length

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=20)</summary>

>

</details>

### bulletize

タグ: 整形

プロンプト:

> 次の文章を、重複を除いた箇条書きに整形してください。各項目は40文字以内、
> 全体で5項目以内にしてください。
>
> バックアップは毎日取得する。取得したバックアップは別の場所へ保管する。
> 復元手順は文書化しておく。復元手順は定期的に試す。バックアップは毎日取得し、
> その結果を記録する。保管期間は用途に応じて決める。

#### qwen3-14b

- 生成速度: 61.5 t/s (出典: eval) / 完了トークン: 512 / finish_reason: length

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=3)</summary>

> 1. バックアップを毎日取得し結果を記録  
> 2. 保管場所を別の場所に設定  
> 3. 復元手順を文書化  
> 4. 復元手

</details>

#### gpt-oss-20b

- 生成速度: 133.5 t/s (出典: eval) / 完了トークン: 512 / finish_reason: length

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=12)</summary>

>

</details>

#### qwen3-8b

- 生成速度: 105.8 t/s (出典: eval) / 完了トークン: 299 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=21)</summary>

> - 毎日バックアップを取得する  
> - 別の場所に保管する  
> - 復元手順を文書化する  
> - 定期的に復元をテストする  
> - 保管期間は用途に応じる

</details>

### explain_term

タグ: 用語説明

プロンプト:

> 「量子化 (quantization)」という用語を、機械学習の入門者向けに200字程度で
> 説明してください。数式は使わないでください。

#### qwen3-14b

- 生成速度: 61.7 t/s (出典: eval) / 完了トークン: 484 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=4)</summary>

> 量子化（quantization）は、機械学習モデルで使用される数値の精度を下げることで、モデルのサイズや計算量を減らす技術です。例えば、32ビットの浮動小数点数を8ビットの整数に変換します。これにより、モデルをスマートフォンや組み込みデバイスなど、リソースが限られた環境でも効率的に動作させることができます。ただし、精度の低下を防ぐため、適切な方法で行う必要があります。

</details>

#### gpt-oss-20b

- 生成速度: 133.5 t/s (出典: eval) / 完了トークン: 512 / finish_reason: length

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=13)</summary>

>

</details>

#### qwen3-8b

- 生成速度: 105.7 t/s (出典: eval) / 完了トークン: 345 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=22)</summary>

> 量子化は、機械学習モデルの数値を低精度に変換する技術です。高い精度のデータを、少ないビット数で表現することで、モデルのサイズを小さくし、計算効率を高めます。これにより、リソースが限られたデバイスでの利用が可能になります。

</details>

### translate_short

タグ: 翻訳

プロンプト:

> 次の日本語を英語に翻訳してください。訳文だけを出力してください。
>
> 「この設定ファイルを変更したら、必ずテストを実行してから反映してください。」

#### qwen3-14b

- 生成速度: 61.8 t/s (出典: eval) / 完了トークン: 512 / finish_reason: length

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=5)</summary>

>

</details>

#### gpt-oss-20b

- 生成速度: 134.4 t/s (出典: eval) / 完了トークン: 159 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=14)</summary>

> If you change this configuration file, be sure to run the tests before applying the changes.

</details>

#### qwen3-8b

- 生成速度: 106.3 t/s (出典: eval) / 完了トークン: 420 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=23)</summary>

> After modifying this configuration file, always run the tests before applying the changes.

</details>

### extract_structured

タグ: 構造化抽出

system:

> あなたは指定された形式を厳密に守るアシスタントです。

プロンプト:

> 次の文から、日時・場所・持ち物を抽出し、JSON オブジェクト1つだけを
> 出力してください。説明文は出力しないでください。
> キーは datetime, place, items の3つとします。
>
> 「次回の勉強会は4月12日の午後2時から、市民会館の第3会議室で行います。
> 参加者はノートパソコンと電源アダプタを持参してください。」

#### qwen3-14b

- 生成速度: 61.5 t/s (出典: eval) / 完了トークン: 512 / finish_reason: length

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=6)</summary>

> {"datetime":"4月12日 午後2時","place":"市民会館 第3

</details>

#### gpt-oss-20b

- 生成速度: 133.8 t/s (出典: eval) / 完了トークン: 314 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=15)</summary>

> {"datetime":"4月12日午後2時から","place":"市民会館の第3会議室","items":["ノートパソコン","電源アダプタ"]}

</details>

#### qwen3-8b

- 生成速度: 105.4 t/s (出典: eval) / 完了トークン: 512 / finish_reason: length

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=24)</summary>

> {
>   "datetime": "4月12日 午後2時",
>   "place": "市民会館 第

</details>

### follow_instructions

タグ: 指示追従

プロンプト:

> 以下の制約をすべて満たして回答してください。
>
> 1. 「バージョン管理を使う利点」を3つ挙げる
> 2. 各利点は必ず「・」で始める
> 3. 各利点は30文字以内にする
> 4. 3つ以外は何も出力しない

#### qwen3-14b

- 生成速度: 62.0 t/s (出典: eval) / 完了トークン: 246 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=7)</summary>

> ・協力作業の容易化  
> ・変更履歴の明確化  
> ・誤操作の修正可能

</details>

#### gpt-oss-20b

- 生成速度: 133.9 t/s (出典: eval) / 完了トークン: 444 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=16)</summary>

> ・作業の履歴を追える  
> ・コラボレーションが容易  
> ・変更のロールバックが可能

</details>

#### qwen3-8b

- 生成速度: 106.0 t/s (出典: eval) / 完了トークン: 231 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=25)</summary>

> ・変更履歴を追跡可能  
> ・多人数での協働が可能  
> ・過去バージョンへの戻り可能

</details>

### ambiguous_request

タグ: 曖昧な依頼

プロンプト:

> 「あれ、いい感じにしておいて。」
>
> この依頼は情報が不足しています。作業に取りかかる前に確認すべきことを、
> 質問の形で3つ挙げてください。推測で作業を始めないでください。

#### qwen3-14b

- 生成速度: 61.7 t/s (出典: eval) / 完了トークン: 359 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=8)</summary>

> 1. **この依頼の対象となる具体的な作業内容やプロジェクトの種類はどのようなものですか？**（例：デザイン、文書作成、プレゼンテーションなど）  
> 2. **対象となる相手（対象者）や、この作業を通じて達成したい目的（意図）は何か教えていただけますか？**  
> 3. **特に気をつけるべきスタイル、フォーマット、または含まれるべき要素など、ご希望の詳細はありますか？**

</details>

#### gpt-oss-20b

- 生成速度: 133.9 t/s (出典: eval) / 完了トークン: 506 / finish_reason: stop

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=17)</summary>

> 1. 「いい感じにしておいて」という指示は、どのようなコンテンツ（文章、画像、動画など）に対してのものですか？  
> 2. 完成したものの具体的な目的やターゲットオーディエンスは何ですか？  
> 3. 納品形式（ファイル形式・サイズ・配布方法など）や締め切りはいつですか？

</details>

#### qwen3-8b

- 生成速度: 105.6 t/s (出典: eval) / 完了トークン: 512 / finish_reason: length

<details>
<summary>応答 (全文 / records.jsonl の sequence_index=26)</summary>

> 以下のように、作業を開始する前に確認すべき3つの質問を挙げます。推測ではなく、明確な情報を得るために必要な点を問うものです。
>
> 1. **「いい感じ」という具体的内容はどのようなものでしょうか？**  
>    （例：デザインの完成度、文章の表現、製品の状態など、明確な基準を教えてください）
>
> 2. **対象となる物・事象はどのようなものでしょうか？**  
>    （例：文書、製品、データ、画像など、対象を特定していただけますか）
>
> 3. **「しておいて」の意味合いは、最終的な完成形を待つことですか？それとも途中の段階を指していますか？**  
>    （例：作業を中断してその状態を保つのか、最終的に完成させるのか、 clarification が必要です）
>
> これらの質問を通

</details>
