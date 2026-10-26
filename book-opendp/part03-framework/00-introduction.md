# The framework

Everything the Context API did in the previous parts was built from a small number of pieces, and this part opens the box. OpenDP's framework describes a private computation as a chain of two kinds of object. A transformation maps data to data, and it promises how much its output can change when its input changes, which is its stability. A measurement maps data to a random release, and it promises how much the distribution of that release can change, which is its privacy. Chaining the two multiplies the promises through, and composing measurements adds them up. Once you can read these objects, you can see exactly what guarantee any query carries. You can also build queries the Context API does not offer, and check a claim someone else makes.

[A framework to understand differential privacy](08-a-framework-for-privacy.md) introduces the vocabulary, namely domains, metrics and measures, and the two maps, the stability map and the privacy map. [Transformations](09-transformations.md) builds the preprocessing and aggregation steps, from clamping and resizing to sums, means and quantile scores. [Measurements](10-measurements.md) covers the mechanisms that add noise or select, from Laplace and Gaussian to thresholded noise, noisy max and randomized response. [Combinators](11-combinators.md) chains and composes them, converts between privacy measures, and shows the adaptive compositors and odometers behind a context. [Utilities and the Context API inside](12-utilities-and-the-context-api.md) covers the type descriptors, the parameter search that finds a scale from a budget, and serialization. It ends by showing how the Context API maps onto the framework.

| Chapter | Question it answers |
| --- | --- |
| [A framework to understand differential privacy](08-a-framework-for-privacy.md) | What are domains, metrics, measures, transformations and measurements, and what does each promise? |
| [Transformations](09-transformations.md) | How do I build and chain the data-to-data steps, and how does stability flow through them? |
| [Measurements](10-measurements.md) | Which mechanisms does OpenDP offer, and how do I calibrate each one? |
| [Combinators](11-combinators.md) | How do I compose measurements, convert between privacy measures, and account adaptively? |
| [Utilities and the Context API inside](12-utilities-and-the-context-api.md) | How do types, parameter search and serialization work, and what does a context do under the hood? |

: The chapters of this part and the questions they answer. {#tbl-p3-chapters}

By the end of this part you will be able to build a private query from its pieces and read off the guarantee of any chain with `map` and `check`. You will also be able to explain what the Context API does when it resolves a query.
