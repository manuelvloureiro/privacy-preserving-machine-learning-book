# The framework

The Context API queries in the previous parts use OpenDP framework components. A private computation chains transformations and measurements. A transformation maps data to data, and its stability bounds how much the output changes when the input changes. A measurement maps data to a random release, and its privacy map bounds changes to the release distribution. These bounds propagate through a chain, while composed measurements accumulate privacy loss. This framework lets you inspect guarantees, build queries unavailable through the Context API and verify claims.

The first chapter defines domains, metrics, measures and the maps that describe stability and privacy. The next three cover transformations, measurements that add noise or select, and combinators for chaining and composition. The final chapter covers supporting utilities and shows how the Context API uses these components.

| Chapter | Question it answers |
| --- | --- |
| [A framework to understand differential privacy](08-a-framework-for-privacy.md) | What are domains, metrics, measures, transformations and measurements, and what does each promise? |
| [Transformations](09-transformations.md) | How do I build and chain the data-to-data steps, and how does stability flow through them? |
| [Measurements](10-measurements.md) | Which mechanisms does OpenDP offer, and how do I calibrate each one? |
| [Combinators](11-combinators.md) | How do I compose measurements, convert between privacy measures, and account adaptively? |
| [Utilities and the Context API inside](12-utilities-and-the-context-api.md) | How do types, parameter search and serialization work, and what does a context do under the hood? |

: The chapters of this part and the questions they answer. {#tbl-p3-chapters}

By the end of this part, you can build private queries from framework components, inspect guarantees with `map` and `check`, and describe how the Context API resolves a query.
