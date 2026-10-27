# Deployments

The earlier parts built differential privacy (DP) with a trusted curator who holds the data and releases noisy answers. This part asks what changes when the definition meets real systems with millions of users, daily data collection and outside analysts. The first chapter removes the curator, so that each person randomizes their own data, and measures the price in error. The second finds a middle ground with a shuffler, then returns to the central model to see how large deployments select, compose and budget.

| Chapter | Question it answers |
| --- | --- |
| [Local differential privacy](01-local-dp.md) | What does it cost to trust no one, and how do large deployments live with that cost? |
| [The shuffle model and central deployments](02-shuffle-and-central-deployments.md) | How much does an anonymizing shuffler recover, and how do central deployments select, compose and budget? |

: The chapters of this part. {#tbl-p5-chapters}
