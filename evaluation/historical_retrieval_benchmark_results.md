# Historical Retrieval Benchmark Results

Saved result from the original Telecom Egypt corpus and development environment. This historical run was not repeated for the fictional sample corpus.

- Cases: **8**
- Passed: **8/8**
- Pass rate: **100.0%**
- Average top-1 score: **0.8779**
- Average retrieval latency: **0.031s**
- Minimum score threshold: **0.80**

| ID | Lang | Query | Normalized | Top source | Score | Latency | Expected entity | Result |
|---|---|---|---|---|---:|---:|---|---|
| AR_WE_SPACE | ar | باقات النت الأرضي من WE؟ | باقات الإنترنت المنزلي من WE؟ | باقات WE Home Internet \| إنترنت منزلي وواي فاي منزلي في مصر - Telecom Egypt | 0.9049 | 0.052s | PASS | PASS |
| EN_WE_SPACE | en | What are WE SPACE packages? | What are WE SPACE packages? | WE Home Internet \| Home Internet, Wi-Fi & Broadband Packages in Egypt - Telecom Egypt | 0.8670 | 0.030s | PASS | PASS |
| EGY_FIXED_INTERNET | ar-eg | إيه باقات النت الارضي؟ | ما هي باقات الإنترنت المنزلي؟ | باقاتOneHome \| خط أرضى مع إنترنت منزلى - Telecom Egypt | 0.8659 | 0.032s | PASS | PASS |
| AR_EKYC | ar | ما هي خدمة eKYC؟ | ما هي خدمة eKYC؟ | الأسئلة الشائعة – التحقق الرقمي (eKYC) - Telecom Egypt | 0.9008 | 0.029s | PASS | PASS |
| EN_EKYC | en | What is eKYC? | What is eKYC? | eKYC FAQs - Telecom Egypt | 0.9105 | 0.027s | PASS | PASS |
| AR_WE_LIFE | ar | ما هي خدمة WE Life؟ | ما هي خدمة WE Life؟ | باقات WE Life \| إنترنت منزلي وترفيه IPTV في فاتورة واحدة بمصر - Telecom Egypt | 0.8678 | 0.027s | PASS | PASS |
| EN_WE_LIFE | en | What is WE LIFE? | What is WE LIFE? | WE Home Internet \| Home Internet, Wi-Fi & Broadband Packages in Egypt - Telecom Egypt | 0.8592 | 0.029s | PASS | PASS |
| AR_WE_SONIC | ar | ما هي WE SONIC؟ | ما هي WE SONIC؟ | باقات WE Sonic \| إنترنت الألعاب السريع وواي فاي منزلي في مصر - Telecom Egypt | 0.8468 | 0.026s | PASS | PASS |

## Interpretation

- `Score` is the top retrieved vector/reranked score returned by the current retrieval stack.
- `Expected entity` checks whether the top retrieved payload contains the expected official product/entity name.
- `Result` requires a non-empty result, score >= threshold, and expected entity match.
- This benchmark evaluates retrieval only. ASR, generation, OCR, and TTS latency should be reported separately.

## Recommended submission metrics

Use this report together with:

- ASR processing time from the notebook.
- LLM generation time from the Streamlit Details section.
- OCR ingestion success examples.
- Piper vs VoiceTut synthesis time.

Do not present a metric as measured unless it was actually produced by the current run.