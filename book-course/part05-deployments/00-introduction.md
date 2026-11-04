# Deployments

The earlier parts developed differential privacy (DP) with a trusted curator who holds the data and releases noisy answers. This part studies deployments with many users, repeated data collection, and external analysts. The first chapter removes the curator: each person randomizes their own data, and the chapter quantifies the resulting error. The second introduces the shuffle model, then examines selection, composition, and budget allocation in central deployments.

| Chapter | Question it answers |
| --- | --- |
| [Local differential privacy](01-local-dp.md) | What does it cost to trust no one, and how do large deployments live with that cost? |
| [The shuffle model and central deployments](02-shuffle-and-central-deployments.md) | How much does an anonymizing shuffler recover, and how do central deployments select, compose and budget? |

: The chapters of this part. {#tbl-p5-chapters}
