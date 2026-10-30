# Privacy failures

Privacy is easy to promise and hard to deliver, so this part shows what goes wrong without a definition before the course gives one. The first chapter looks at real releases that their owners protected with pseudonyms, removed names, summary statistics or a trained model. An attacker defeated each one by combining the release with information from another source. The second chapter turns these cases into theorems: answers to counting queries that are too accurate let an analyst rebuild almost every secret bit. Together the chapters set the requirements for a definition of privacy. It must hold against any side information and apply to any kind of output. It must degrade gracefully as more data comes out, and it must allow the error that the attacks show to be necessary.

| Chapter | Question it answers |
| --- | --- |
| [Attempts at Data Privacy](01-attempts.md) | Why do pseudonyms, removing identifiers, releasing models or statistics, and $k$-anonymity fail to protect people? |
| [Reconstruction Attacks](02-reconstruction.md) | How much error must answers to counting queries have before an analyst cannot rebuild the data? |

: The chapters of this part. {#tbl-p1-chapters}
