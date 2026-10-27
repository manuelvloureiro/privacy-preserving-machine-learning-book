# The framework

Every query that the Context API ran in the previous parts was built from a few pieces, and this part shows those pieces. OpenDP's framework describes a private computation as a chain of two kinds of object. A transformation maps data to data, and its stability bounds how much its output can change when its input changes. A measurement maps data to a random release, and its privacy bounds how much the distribution of the release can change. In a chain, these promises pass from one step to the next, and when you compose measurements, their privacy losses add up. So when you can read these objects, you can see exactly which guarantee a query carries. You can also build queries that the Context API does not offer, and check a claim that someone else makes.

The first chapter gives the vocabulary: domains, metrics and measures, and the two maps that carry the promises. The next three build the pieces in order: the data-to-data steps, the mechanisms that add noise or select, and the combinators that chain and compose them. The last chapter covers the tools around the pieces and shows how the Context API maps onto them.

| Chapter | Question it answers |
| --- | --- |
| [A framework to understand differential privacy](08-a-framework-for-privacy.md) | What are domains, metrics, measures, transformations and measurements, and what does each promise? |
| [Transformations](09-transformations.md) | How do I build and chain the data-to-data steps, and how does stability flow through them? |
| [Measurements](10-measurements.md) | Which mechanisms does OpenDP offer, and how do I calibrate each one? |
| [Combinators](11-combinators.md) | How do I compose measurements, convert between privacy measures, and account adaptively? |
| [Utilities and the Context API inside](12-utilities-and-the-context-api.md) | How do types, parameter search and serialization work, and what does a context do under the hood? |

: The chapters of this part and the questions they answer. {#tbl-p3-chapters}

At the end of this part, you can build a private query from its pieces and read the guarantee of any chain with `map` and `check`. You can also explain what the Context API does when it resolves a query.
