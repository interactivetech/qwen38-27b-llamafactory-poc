# Output-quality screen: base versus trained adapter

This is an exploratory, non-executing comparison on four existing validation contexts plus four controlled instruction/format probes. Both models received identical input tokens, greedy decoding, the `qwen3_8` template with `xhigh` reasoning effort, and up to 2,048 new tokens. No responses were repaired or regenerated to improve the scores. The held-out reference next responses were kept out of the model inputs.

## Measured results

| Check | Base | Adapter |
| --- | --- | --- |
| Complete, schema-valid responses on real contexts | 1/4 | 4/4 |
| Real-context responses reaching the token cap | 3/4 | 0/4 |
| Exact-action instruction probes passed | 4/4 | 4/4 |
| Real task executions | 0 | 0 |

Base held-out output lengths were **226, 2048, 2048, 2048** tokens. Adapter lengths were **372, 1202, 1635, 1699** tokens. All uncapped outputs in this screen passed the structural checks. Capped base outputs had not delivered final JSON within the budget; this is a completion/budget issue, not evidence of invalid syntax in a finished answer.

The four controlled probes explicitly specified the correct next action: inspect a file with a supplied command, poll an already running test without launching a duplicate, request an identity diagnostic after permission denial without privilege escalation, and acknowledge completion after a supplied successful verification. Their exact-action pass rates test instruction following and serialization, not independent planning or actual command success.

## Context-based review

The following is an assistant's static review against the supplied observations and reference response. It is not independent human grading or an executable benchmark. Different next actions can be valid; exact reference matching was not used as a quality metric.

| Real context | Base | Adapter | Review and remaining uncertainty |
| --- | --- | --- | --- |
| WAMR memory64 implementation | Complete, 226 tokens | Complete, 372 tokens | Both propose plausible read-only source inspection. The adapter also examines CMake feature definitions. Neither response implements or tests memory64, so implementation correctness is unmeasured. |
| Protected query gateway reload | Capped at 2,048 tokens | Complete, 1,202 tokens | The base response repeats speculation about the reload interface. The adapter proposes a concrete HTTP reload request and policy inspection. Its first request sends the raw candidate; the reference instead wraps it under a `candidate` key. Whether the raw payload is accepted requires checking the actual endpoint. Completion does not prove the action is correct. |
| Adaptive network campaign | Capped at 2,048 tokens | Complete, 1,635 tokens | The adapter proposes direct-edge HTTP controls using edge names and paths present in the supplied inventory, with affected and comparison sources. That is a plausible next diagnostic action; measurement success, causal conclusions, and final campaign completion were not tested. |
| RF occupancy workflow | Capped at 2,048 tokens | Complete, 1,699 tokens | The adapter recognizes center-spur and coverage uncertainty and proposes short capture experiments with flags/units present in the supplied CLI help. Acceptance of those instrument settings, calibration validity, and the final reusable occupancy workflow remain untested. |

## Interpretation

The adapter delivered complete structured next actions more consistently in these four contexts under the fixed output budget. Both models followed the four simple exact-action probes. These results support a narrow improvement in response usability; **4/4 complete JSON responses is not 4/4 tasks solved**.

The held-out reference lengths were **494, 1057, 2891, 6576** tokens. Two exceed the 2,048-token output budget. A capped response is therefore inconclusive about the model's ability with a larger budget, and this screen should not be treated as a general accuracy ranking. The validation contexts were already used by the pilot's evaluation, so this is not a new untouched benchmark set.

The next substantive quality test is to execute a broader set of tasks in their matching isolated environments, grade observable outcomes, and test error recovery and constraints. Full prompts, model responses, and reference comparisons remain in the local quality run as `comparison.md`; the public JSON includes only aggregate/case-level metrics and hashes.
