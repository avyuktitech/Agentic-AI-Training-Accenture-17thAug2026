# Executive Summary -- Agentic Remittance & Payment Matching Program
_Generated: 2026-08-19 19:52 UTC_

## 1. Volume & Straight-Through Processing
- Documents ingested: 5
- Straight-through rate: 60%
- Payments resolved: 5
- Confident-match rate (>=0.6 conf): 80%
- Net matching variance (short-pay/over-pay): -1,200.00

## 2. Agent Maturity
| agent                         |   overall_score | maturity_level   |
|:------------------------------|----------------:|:-----------------|
| Capstone1_RemittanceIngestion |             3.4 | 3 - Defined      |
| Capstone2_PaymentMatching     |             4.2 | 4 - Managed      |

## 3. Top Risks (by heat score)
| risk_id   | agent                         | category    | description                                                                |   heat_score |
|:----------|:------------------------------|:------------|:---------------------------------------------------------------------------|-------------:|
| RSK-003   | Capstone2_PaymentMatching     | model       | Smart-match model trained on small synthetic set may not generalize        |           12 |
| RSK-001   | Capstone1_RemittanceIngestion | data        | Malformed/incomplete ERP rows silently mis-extracted                       |            9 |
| RSK-004   | Capstone2_PaymentMatching     | operational | Partial-payment short-pay variance not routed to collections automatically |            9 |

## 4. Responsible AI Governance Flags
| check            | record_id   | detail                                                                                              |
|:-----------------|:------------|:----------------------------------------------------------------------------------------------------|
| segment_fairness | ALL         | Auto-approval rate spread across segments = 50%: {'Enterprise': 0.5, 'Mid-Market': 1.0, 'SMB': 1.0} |

## 5. ERP Posting Status
| journal_id   | status   |   lines_posted | posted_at                  |
|:-------------|:---------|---------------:|:---------------------------|
| JE-PMT-01    | POSTED   |              2 | 2026-08-19T19:52:53.619464 |
| JE-PMT-02    | POSTED   |              2 | 2026-08-19T19:52:53.620240 |
| JE-PMT-03    | POSTED   |              2 | 2026-08-19T19:52:53.620865 |
| JE-PMT-04    | POSTED   |              3 | 2026-08-19T19:52:53.621657 |
| JE-PMT-05    | POSTED   |              2 | 2026-08-19T19:52:53.622328 |

## 6. Recommendation
- Promote Capstone 2 (Payment Matching) toward pilot: highest maturity score, full observability and test coverage.
- Hold Capstone 1 (Remittance Ingestion) at current stage until rollback readiness and observability dashboards are improved (see RSK-002).
- Route RSK-003 and RSK-005 to the governance board given heat scores >= 8.