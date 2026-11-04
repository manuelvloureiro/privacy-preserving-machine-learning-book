# Extending the library

OpenDP's framework accepts built-in and user-defined transformations and measurements with stability or privacy maps. A plugin preserves privacy only if its map is valid; you must prove that validity. The framework then supports chaining and accounting.

[Plugins](13-plugins.md) implements a transformation and measurement in Python, registers Context API constructors, and reproduces the documentation's examples: Theil-Sen regression and private selection of grouping columns. [Statistical modelling](14-statistical-modelling.md) fits differentially private linear regression and principal component analysis models and interprets their guarantees. [Limitations and troubleshooting](15-limitations-and-troubleshooting.md) covers limitations such as floating-point arithmetic and side channels, feature flags and common errors.

| Chapter | Question it answers |
| --- | --- |
| [Plugins](13-plugins.md) | How do I add my own transformation or measurement, and what must I prove about it? |
| [Statistical modelling](14-statistical-modelling.md) | How do I fit a private regression or a private principal component analysis with the library's estimators? |
| [Limitations and troubleshooting](15-limitations-and-troubleshooting.md) | What does the library not guarantee, and what does each common error mean? |

: The chapters of this part and the questions they answer. {#tbl-p4-chapters}

By the end of this part, you can implement a mechanism, fit the built-in models and identify your remaining deployment responsibilities.
