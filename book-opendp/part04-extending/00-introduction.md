# Extending the library

The framework is open. Anything with a stability map or a privacy map can be a transformation or a measurement, whether OpenDP wrote it or you did. The thread of this part is that a plugin is only as private as its map. So writing one means proving its map, and the library makes the rest safe, from chaining to accounting.

[Plugins](13-plugins.md) writes a transformation and a measurement in Python and registers constructors for the Context API. Then it follows the documentation's two larger examples, a Theil-Sen regression and a private choice of grouping columns. [Statistical modelling](14-statistical-modelling.md) fits the two models that come with the library, a differentially private linear regression and a differentially private principal component analysis, and reads what each guarantees. [Limitations and troubleshooting](15-limitations-and-troubleshooting.md) says what OpenDP does not promise, from floating-point arithmetic to side channels. It also explains the feature flags and the common error messages.

| Chapter | Question it answers |
| --- | --- |
| [Plugins](13-plugins.md) | How do I add my own transformation or measurement, and what must I prove about it? |
| [Statistical modelling](14-statistical-modelling.md) | How do I fit a private regression or a private principal component analysis with the library's estimators? |
| [Limitations and troubleshooting](15-limitations-and-troubleshooting.md) | What does the library not guarantee, and what does each common error mean? |

: The chapters of this part and the questions they answer. {#tbl-p4-chapters}

At the end of this part you can add a mechanism of your own, fit the built-in models, and say which parts of a deployment stay yours.
