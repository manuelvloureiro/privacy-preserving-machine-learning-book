# Privacy failures

This part examines privacy failures before defining privacy. Its first chapter shows how attackers combine releases protected by pseudonyms, identifier removal, summary statistics, or model training with information from other sources. Its second proves that accurate answers to counting queries can reveal almost every secret bit. These failures motivate a definition that allows arbitrary side information, covers every output, degrades gracefully under repeated releases, and permits enough error to prevent reconstruction.

| Chapter | Question it answers |
| --- | --- |
| [Attempts at Data Privacy](01-attempts.md) | Why do pseudonyms, removing identifiers, releasing models or statistics, and $k$-anonymity fail to protect people? |
| [Reconstruction Attacks](02-reconstruction.md) | How much error must answers to counting queries have before an analyst cannot rebuild the data? |

: The chapters of this part. {#tbl-p1-chapters}
