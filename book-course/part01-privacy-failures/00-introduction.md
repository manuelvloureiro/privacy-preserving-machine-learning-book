# Privacy Failures

Privacy is easy to promise and hard to deliver. Before the course defines differential privacy, this part shows what goes wrong without a definition. Its two chapters look at the problem from opposite sides, first through real releases that failed, then through theorems that say exactly when a release must fail.

The first chapter goes through five releases of sensitive data whose owners took care to protect them. They replaced identifiers by pseudonyms, published only a trained model or only summary statistics, or removed names. Every protection was defeated, and almost always in the same way. The attacker combined the release with information from another source, such as a public register, a review website or a voter roll. No protection chosen by intuition said what it guaranteed against such side information. The chapter ends with $k$-anonymity, which coarsens records until each looks like several others, and shows that it fails when two releases are combined.

The second chapter turns these anecdotes into mathematics. It asks how much an analyst can learn from counting queries about a dataset of $n$ secret bits when every answer may be wrong by a bounded amount. If all possible queries are answered, errors must be of order $n$ to stop the analyst from rebuilding almost every bit. If only about $n$ random queries are answered, errors of order $\sqrt n$ are needed, and an efficient attack based on linear programming does the rebuilding. The same attack broke a deployed query system whose answers were too accurate.

Together the chapters set the requirements for a definition of privacy. It must hold against any side information, apply to any kind of output, and degrade gracefully as more is released. It must also allow enough error, which the reconstruction attacks show cannot be avoided. Differential privacy, defined in the next part, is built to meet these requirements.

| Chapter | Question it answers |
| --- | --- |
| [Attempts at Data Privacy](01-attempts.md) | Why do pseudonyms, removing identifiers, releasing models or statistics, and $k$-anonymity fail to protect people? |
| [Reconstruction Attacks](02-reconstruction.md) | How much error must answers to counting queries have before an analyst cannot rebuild the data? |

: The chapters of this part. {#tbl-p1-chapters}
