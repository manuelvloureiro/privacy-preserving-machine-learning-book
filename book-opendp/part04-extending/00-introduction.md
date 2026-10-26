# Extending the library

The framework is open. Anything with a stability map or a privacy map can be a transformation or a measurement, whether OpenDP wrote it or you did. This part uses that openness. Its thread is that a plugin is only as private as its map. So the work of writing one is the proof, and the library's job is to make everything around the proof safe, from chaining to accounting.

[Plugins](13-plugins.md) writes a transformation and a measurement in Python and registers constructors so that they work from the Context API. It then follows the documentation's two larger examples, a Theil-Sen regression and a private choice of grouping columns. [Statistical modelling](14-statistical-modelling.md) fits the two models that ship with the library. One is a differentially private linear regression, and the other a differentially private principal component analysis. The chapter reads what each guarantees. [Limitations and troubleshooting](15-limitations-and-troubleshooting.md) says what OpenDP does not promise, from floating-point arithmetic to side channels. It explains the feature flags that gate the less vetted parts, and it decodes the error messages you will meet most often.

| Chapter | Question it answers |
| --- | --- |
| [Plugins](13-plugins.md) | How do I add my own transformation or measurement, and what must I prove about it? |
| [Statistical modelling](14-statistical-modelling.md) | How do I fit a private regression or a private principal component analysis with the library's estimators? |
| [Limitations and troubleshooting](15-limitations-and-troubleshooting.md) | What does the library not guarantee, and what does each common error mean? |

: The chapters of this part and the questions they answer. {#tbl-p4-chapters}

By the end of this part you will be able to extend OpenDP with a mechanism of your own and fit its built-in models. You will also be able to say which parts of a deployment the library covers and which parts remain yours.
