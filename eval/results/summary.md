# Day 20 DEV comparison

Each configuration uses the same 45 DEV questions. Grounded retrieval uses k=5.

## Overall

| Configuration | Questions | Recall@5 | Correctness | Faithfulness | Refusal accuracy | Over-refusal | p50 ms | p95 ms | Mean tokens | USD/query | Cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `closed_book` | 45 | N/A | 28/36 | N/A | 9/9 | 3/36 | 1,981.21 | 10,456.98 | 96.96 | $0.00 | 8/45 |
| `dense_fixed` | 45 | 32/36 | 24/36 | 24/24 | 9/9 | 12/36 | 1,480.88 | 2,189.32 | 1,638.84 | $0.00 | 20/45 |
| `dense_heading` | 45 | 34/36 | 27/36 | 27/27 | 9/9 | 9/36 | 2,637.81 | 7,085.60 | 1,696.31 | $0.00 | 0/45 |
| `bm25_heading` | 45 | 35/36 | 30/36 | 30/30 | 9/9 | 6/36 | 1,879.94 | 11,060.69 | 1,632.71 | $0.00 | 6/45 |
| `hybrid_heading` | 45 | 35/36 | 30/36 | 30/30 | 9/9 | 6/36 | 1,745.49 | 2,448.13 | 1,659.91 | $0.00 | 1/45 |
| `hybrid_rerank_heading` | 45 | 35/36 | 29/36 | 29/29 | 9/9 | 7/36 | 3,166.62 | 3,803.53 | 1,691.49 | $0.00 | 1/45 |

## By question type

### Lookup

| Configuration | Questions | Recall@5 | Correctness | Faithfulness | Refusal accuracy | Over-refusal | p50 ms | p95 ms | Mean tokens | USD/query | Cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `closed_book` | 18 | N/A | 16/18 | N/A | N/A | 0/18 | 1,191.99 | 2,294.18 | 86.89 | $0.00 | 8/18 |
| `dense_fixed` | 18 | 17/18 | 15/18 | 15/15 | N/A | 3/18 | 38.40 | 102.30 | 1,669.39 | $0.00 | 18/18 |
| `dense_heading` | 18 | 17/18 | 16/18 | 16/16 | N/A | 2/18 | 2,912.75 | 5,277.03 | 1,614.33 | $0.00 | 0/18 |
| `bm25_heading` | 18 | 18/18 | 18/18 | 18/18 | N/A | 0/18 | 2,065.86 | 13,626.99 | 1,644.50 | $0.00 | 2/18 |
| `hybrid_heading` | 18 | 18/18 | 18/18 | 18/18 | N/A | 0/18 | 1,805.41 | 2,763.89 | 1,610.72 | $0.00 | 0/18 |
| `hybrid_rerank_heading` | 18 | 18/18 | 17/18 | 17/17 | N/A | 1/18 | 3,159.20 | 3,998.84 | 1,687.17 | $0.00 | 1/18 |

### Multi-step

| Configuration | Questions | Recall@5 | Correctness | Faithfulness | Refusal accuracy | Over-refusal | p50 ms | p95 ms | Mean tokens | USD/query | Cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `closed_book` | 9 | N/A | 8/9 | N/A | N/A | 0/9 | 2,087.83 | 2,768.41 | 170.78 | $0.00 | 0/9 |
| `dense_fixed` | 9 | 7/9 | 1/9 | 1/1 | N/A | 8/9 | 1,980.75 | 2,432.56 | 1,621.11 | $0.00 | 2/9 |
| `dense_heading` | 9 | 8/9 | 2/9 | 2/2 | N/A | 7/9 | 6,416.98 | 8,649.52 | 1,742.89 | $0.00 | 0/9 |
| `bm25_heading` | 9 | 8/9 | 3/9 | 3/3 | N/A | 6/9 | 1,854.49 | 3,421.08 | 1,488.89 | $0.00 | 1/9 |
| `hybrid_heading` | 9 | 8/9 | 3/9 | 3/3 | N/A | 6/9 | 1,753.24 | 2,020.03 | 1,618.78 | $0.00 | 0/9 |
| `hybrid_rerank_heading` | 9 | 8/9 | 3/9 | 3/3 | N/A | 6/9 | 3,185.36 | 3,561.87 | 1,625.67 | $0.00 | 0/9 |

### New or changed

| Configuration | Questions | Recall@5 | Correctness | Faithfulness | Refusal accuracy | Over-refusal | p50 ms | p95 ms | Mean tokens | USD/query | Cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `closed_book` | 9 | N/A | 4/9 | N/A | N/A | 3/9 | 2,813.86 | 10,431.62 | 87.78 | $0.00 | 0/9 |
| `dense_fixed` | 9 | 8/9 | 8/9 | 8/8 | N/A | 1/9 | 1,768.23 | 1,962.61 | 1,646.00 | $0.00 | 0/9 |
| `dense_heading` | 9 | 9/9 | 9/9 | 9/9 | N/A | 0/9 | 2,426.92 | 3,352.19 | 1,822.44 | $0.00 | 0/9 |
| `bm25_heading` | 9 | 9/9 | 9/9 | 9/9 | N/A | 0/9 | 1,855.63 | 9,206.95 | 1,758.33 | $0.00 | 1/9 |
| `hybrid_heading` | 9 | 9/9 | 9/9 | 9/9 | N/A | 0/9 | 1,506.76 | 1,956.77 | 1,805.11 | $0.00 | 1/9 |
| `hybrid_rerank_heading` | 9 | 9/9 | 9/9 | 9/9 | N/A | 0/9 | 3,311.67 | 3,723.74 | 1,768.44 | $0.00 | 0/9 |

### Unanswerable

| Configuration | Questions | Recall@5 | Correctness | Faithfulness | Refusal accuracy | Over-refusal | p50 ms | p95 ms | Mean tokens | USD/query | Cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `closed_book` | 9 | N/A | N/A | N/A | 9/9 | N/A | 2,802.74 | 12,999.19 | 52.44 | $0.00 | 0/9 |
| `dense_fixed` | 9 | N/A | N/A | N/A | 9/9 | N/A | 1,876.36 | 2,206.22 | 1,588.33 | $0.00 | 0/9 |
| `dense_heading` | 9 | N/A | N/A | N/A | 9/9 | N/A | 2,593.72 | 4,441.90 | 1,687.56 | $0.00 | 0/9 |
| `bm25_heading` | 9 | N/A | N/A | N/A | 9/9 | N/A | 1,811.37 | 2,130.74 | 1,627.33 | $0.00 | 2/9 |
| `hybrid_heading` | 9 | N/A | N/A | N/A | 9/9 | N/A | 1,640.71 | 2,300.51 | 1,654.22 | $0.00 | 0/9 |
| `hybrid_rerank_heading` | 9 | N/A | N/A | N/A | 9/9 | N/A | 3,022.46 | 3,689.79 | 1,689.00 | $0.00 | 0/9 |

## Metric definitions

- Recall@5 counts answerable grounded questions with at least one gold source page among the saved top five passages. It is inapplicable to closed-book.

- Correctness includes all answerable questions. Exact abstentions score zero.

- Faithfulness includes all grounded answers other than exact refusals. It is inapplicable to closed-book.

- Refusal accuracy uses unanswerable questions. Over-refusal uses answerable questions. Both use the configuration's exact abstention string.

- N/A means the metric has no eligible rows. The CSV uses an empty numerator and a zero denominator for these cells.

## Timing, tokens, and cost

- Latency is saved retrieval time plus saved LLM call time, using inclusive percentiles. Cache hits include lookup time. Saved rows span different runs and quota windows, so these figures describe recorded runs.

- Mean tokens use original generation usage, including input, output, and thinking tokens. Cached answers retain the original completion's token counts.

- Generation cost per query is $0 on the billing-disabled Free tier. Judge requests are evaluation overhead. Fresh completion attempts also make a separate token-count request.

- Cache-hit counts describe the invocation that saved each file. Requests completed before a restart can appear as cache hits in the final file. Saved API-usage fields cover successful completions in the saving invocation.
