# Deployments

The earlier parts built differential privacy (DP) as mathematics, with a trusted curator who holds the data and releases noisy answers. This part asks what happens when the definition meets real systems, with millions of users, daily data collection and analysts outside the company. The deployments studied here are not just applications of earlier results. Each forced a new idea, and most of those ideas turned into theory of their own.

The first chapter removes the trusted curator. In the local model each person randomizes their own data before sending it, so no one ever sees a raw record. The price is an error about $\sqrt n$ times larger than in the central model for $n$ people, and a lower bound shows that no noninteractive protocol can avoid it. Google, Apple and Microsoft accepted this price because they have hundreds of millions of users, and each solved a practical problem along the way. They had to handle questions repeated every day, find values that no one knew to ask about, and send numbers in a single bit.

The second chapter looks for a middle ground and then returns to the central model. A shuffler that only mixes messages lets each person add far less noise, and a sum of bits can then be estimated with an error that does not grow with $n$. A query service for advertisers brings a sharper analysis of the exponential mechanism and a fast way to select the top items of a histogram. Two large public data releases show how deployments choose what a neighbouring dataset is, and how they account for privacy with zero-concentrated differential privacy.

| Chapter | Question it answers |
| --- | --- |
| [Local differential privacy](01-local-dp.md) | What does it cost to trust no one, and how do large deployments live with that cost? |
| [The shuffle model and central deployments](02-shuffle-and-central-deployments.md) | How much does an anonymizing shuffler recover, and how do central deployments select, compose and budget? |

: The chapters of this part. {#tbl-p5-chapters}
