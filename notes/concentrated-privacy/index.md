---
title: "From Differential Privacy to Concentrated Privacy"
subtitle: "A source-guided derivation of definitions, bounds, and composition"
author: "Technical study notes"
date: "23 October 2026"
abstract: |
  This note develops pure $\varepsilon$-DP, approximate $(\varepsilon,\delta)$-DP, mean-concentrated DP (CDP/mCDP), and zero-concentrated DP (zCDP). It follows the mathematical progression in Dwork–Rothblum–Vadhan (2010), Dwork–Rothblum (2016), and Bun–Steinke (2016). Every major formula is preceded by the question it answers. Elementary derivations are given in full, including the algebra and the probabilistic lemmas they rest on; non-elementary results are named and cited, not silently assumed.
---

# The common mathematical object: two output distributions {#sec-object}

Take neighbouring datasets $x\sim x'$ that differ in one person's record, and one randomized mechanism $M$. Write $P=\mathcal L(M(x))$ and $Q=\mathcal L(M(x'))$ for its two output distributions, which are probability measures. For an event $S$, $P(S)$ and $Q(S)$ are probabilities. If the two measures have densities $p,q$ with respect to a common dominating measure $\nu$, then

$$P(S)=\int_S p(y)\,\mathrm d\nu(y),\qquad Q(S)=\int_S q(y)\,\mathrm d\nu(y).$$

The measure $\nu$ is Lebesgue measure in continuous examples and counting measure in discrete ones (one can always take $\nu=P+Q$). The notation therefore never mistakes a density value for the probability of an individual output.

Whenever $P\ll Q$, define the likelihood ratio $R=\mathrm dP/\mathrm dQ$ and the **privacy loss** $L(y)=\log R(y)$. It is defined $Q$-almost surely; when we sample $Y\sim P$, $L(Y)$ is a random variable. Under pure DP, $P$ and $Q$ are mutually absolutely continuous, so the ratio is finite and bounded almost surely. Under approximate DP this need not be true: $P$ may put mass where $Q$ has none, and there we set $L=+\infty$.

**Why almost surely?** Suppose $P(S)\le e^\varepsilon Q(S)$ holds for every measurable event. Then $p(y)\le e^\varepsilon q(y)$ for $\nu$-almost every $y$, though not necessarily at every point of an arbitrarily chosen version of the densities. The argument runs in both directions:

* If $p\le e^\varepsilon q$ holds $\nu$-almost everywhere, integrating over $S$ gives $P(S)\le e^\varepsilon Q(S)$ exactly, because a null set contributes nothing to an integral.
* If $p>e^\varepsilon q$ on a set $B$ with $\nu(B)>0$, then $\int_B(p-e^\varepsilon q)\,\mathrm d\nu>0$ (a strictly positive function integrated over a set of positive measure), so $P(B)>e^\varepsilon Q(B)$ and the event $S=B$ breaks the inequality.

Whenever a density-ratio bound is used below, its scope is stated.

# Pure $\varepsilon$-differential privacy {#sec-pure}

## Definition and interpretation

The mechanism is $\varepsilon$-DP ($\varepsilon\ge0$) when, for every ordered pair $x\sim x'$ and every measurable $S$,

$$P(S)\le e^\varepsilon Q(S). \tag{2.1}$$

Since the neighbour relation is symmetric, the definition also applies to $(x',x)$, so $Q(S)\le e^\varepsilon P(S)$ holds too. By the argument of @sec-object applied in both orders, the equivalent likelihood-ratio condition is

$$e^{-\varepsilon}\le R(y)\le e^\varepsilon\quad Q\text{-a.s.},\qquad -\varepsilon\le L(y)\le\varepsilon\quad P\text{-a.s.} \tag{2.2}$$

(The two statements agree because $P$ and $Q$ have the same null sets.) The guarantee bounds every event probability, not only typical events. At $\varepsilon=0$, $P=Q$: the outputs are identically distributed on neighbouring inputs.

## Why basic composition adds $\varepsilon$

**What is being composed.** An adaptive analyst runs mechanisms $M_1,\dots,M_k$ in turn; the $i$-th may depend on the earlier outputs, $Y_i\sim M_i(x,y_{<i})$. The analyst's choices are the same function of the transcript whichever dataset is in use; only the data differ. The hypothesis is that for every fixed history $y_{<i}$ the mechanism $M_i(\cdot,y_{<i})$ is $\varepsilon_i$-DP. Write $p_i(\cdot\mid y_{<i})$ and $q_i(\cdot\mid y_{<i})$ for its output densities on $x$ and $x'$.

The density of the whole transcript is the product of the conditional densities (the chain rule), on each dataset:

$$\frac{p(y_{1:k})}{q(y_{1:k})}=\prod_{i=1}^k\frac{p_i(y_i\mid y_{<i})}{q_i(y_i\mid y_{<i})},\qquad\text{so}\qquad L_{1:k}=\sum_{i=1}^kL_i .$$

Each factor is a likelihood ratio of an $\varepsilon_i$-DP mechanism with its history fixed, so by (2.2) $|L_i|\le\varepsilon_i$ almost surely, and hence $|L_{1:k}|\le\sum_i\varepsilon_i$ almost surely. Integrating that density bound over an event (@sec-object) gives the event inequality, so adaptive composition is $(\sum_i\varepsilon_i)$-DP. In particular, $k$ identical guarantees give $k\varepsilon$-DP.

## Post-processing

If $g$ is any function applied to the output, then $\Pr[g(M(x))\in S]=P(g^{-1}(S))\le e^\varepsilon Q(g^{-1}(S))$, because $g^{-1}(S)$ is itself an event. A randomized $g$ is a mixture of deterministic ones, and averaging the inequality over the mixture preserves it. The same one-line argument works for $(\varepsilon,\delta)$-DP.

## Example: Laplace noise and sensitivity

For a numerical query $f$ with sensitivity $\Delta=\sup_{x\sim x'}|f(x)-f(x')|$, release $Y=f(x)+Z$ with $Z\sim\mathrm{Laplace}(0,b)$, whose density is $p_x(y)=(2b)^{-1}\exp(-|y-f(x)|/b)$. The normalizing constants cancel in the ratio, leaving

$$\log\frac{p_x(y)}{p_{x'}(y)}=\frac{|y-f(x')|-|y-f(x)|}{b}\le\frac{|f(x)-f(x')|}{b}\le\frac{\Delta}{b}.$$

The first inequality is the triangle inequality, $|u|-|v|\le|u-v|$ with $u=y-f(x')$ and $v=y-f(x)$. Setting $b=\Delta/\varepsilon$ gives pure $\varepsilon$-DP. This calculation bounds the ratio at every point for this continuous version of the density, which is stronger than the almost-sure bound the definition needs. For a vector-valued $f$, use independent Laplace noise in each coordinate and the $\ell_1$-sensitivity; the same calculation applies coordinate by coordinate and the bounds add.

# Approximate $(\varepsilon,\delta)$-differential privacy {#sec-approx}

## Definition: what $\delta$ actually means

For $\varepsilon\ge0$ and $0\le\delta\le1$, we require, for every neighbouring ordered pair and every event $S$,

$$P(S)\le e^\varepsilon Q(S)+\delta. \tag{3.1}$$

Unlike pure DP, this does not give a likelihood ratio that is bounded almost surely. It allows an additive discrepancy in the probability of an event. It is **not** in general equivalent to "the privacy loss exceeds $\varepsilon$ with probability at most $\delta$" with the same parameters; @sec-tail makes both directions precise.

## The exact meaning of $\delta$: the hockey-stick divergence

Define $H_\varepsilon(P\|Q)=\sup_S[P(S)-e^\varepsilon Q(S)]$. With densities, the supremum is achieved by the set where $p(y)>e^\varepsilon q(y)$, so

$$H_\varepsilon(P\|Q)=\int\bigl(p(y)-e^\varepsilon q(y)\bigr)_+\,\mathrm d\nu(y),\qquad u_+=\max(u,0). \tag{3.2}$$

To see this, write the difference for an event as $\int_S(p-e^\varepsilon q)\,\mathrm d\nu$. Adding to $S$ points where the integrand is negative can only lower it, and adding every point where it is positive raises it to its maximum. Hence $(\varepsilon,\delta)$-DP is exactly $H_\varepsilon(P\|Q)\le\delta$ **and** $H_\varepsilon(Q\|P)\le\delta$ for every neighbouring pair. Both directions are needed because $H_\varepsilon$ is not symmetric. This is the precise operational meaning of $\delta$.

If $P$ has mass that is singular with respect to $Q$ (where $q=0$), that mass counts in full towards $H_\varepsilon$. The loss can therefore be infinite on some outcomes while $(\varepsilon,\delta)$-DP still holds.

## From a tail bound on the loss to $(\varepsilon,\delta)$-DP, and why the converse fails {#sec-tail}

Every conversion below ends with the same step, so we prove it once.

**Lemma 3.1 (tail bound implies approximate DP).** If $\Pr_{Y\sim P}[L(Y)>\varepsilon]\le\eta$, then $P(S)\le e^\varepsilon Q(S)+\eta$ for every event $S$.

*Proof.* Split $S$ into $S_{\le}=S\cap\{L\le\varepsilon\}$ and $S_>=S\cap\{L>\varepsilon\}$. On $S_\le$ the density bound $p\le e^\varepsilon q$ holds (on the part where $q=0$, $L=+\infty$, so that part lies in $S_>$), and integrating it gives $P(S_\le)\le e^\varepsilon Q(S_\le)\le e^\varepsilon Q(S)$. The other part has $P(S_>)\le P(L>\varepsilon)\le\eta$. Add the two. $\square$

The same computation gives an exact formula. On the optimal set $\{L>\varepsilon\}$, $q=e^{-L}p$, so

$$H_\varepsilon(P\|Q)=\mathbb E_P\bigl[(1-e^{\varepsilon-L})_+\bigr]\le\Pr_P[L>\varepsilon]. \tag{3.3}$$

The weight $1-e^{\varepsilon-L}$ is close to $0$ when $L$ only slightly exceeds $\varepsilon$ and close to $1$ when $L$ is far above it. So $\delta$ is the tail probability of the loss *discounted* by how far the loss goes beyond $\varepsilon$.

**The converse fails.** Take $\varepsilon=0$, $P=(\tfrac12,\tfrac12)$ and $Q=(\tfrac12+\delta,\tfrac12-\delta)$ on two outcomes. Then $H_0(P\|Q)=H_0(Q\|P)=\delta$, so the pair is $(0,\delta)$-indistinguishable. Yet $L>0$ on the second outcome, which has probability $\tfrac12$ under $P$, however small $\delta$ is.

## Two probabilistic tools

The concentration arguments below use only these two facts.

**Chernoff's bound.** For any random variable $X$, any $t$ and any $\lambda>0$, Markov's inequality applied to the nonnegative variable $e^{\lambda X}$ gives

$$\Pr[X\ge t]=\Pr[e^{\lambda X}\ge e^{\lambda t}]\le e^{-\lambda t}\,\mathbb E[e^{\lambda X}]. \tag{3.4}$$

**Hoeffding's lemma** (cited, not proved). If $\mathbb E X=0$ and $a\le X\le b$ almost surely, then $\mathbb E[e^{\lambda X}]\le\exp\bigl(\lambda^2(b-a)^2/8\bigr)$ for every real $\lambda$. The same holds conditionally on a $\sigma$-field when the conditional mean is $0$ and the range has width $b-a$. For a range of width $2\varepsilon$ the exponent is $\lambda^2\varepsilon^2/2$.

## Composition bounds

Basic adaptive composition of $(\varepsilon_i,\delta_i)$-DP mechanisms gives $(\sum_i\varepsilon_i,\sum_i\delta_i)$-DP. The advanced composition theorem of Dwork–Rothblum–Vadhan (2010), with the improved mean term of Dwork–Rothblum (2016, Theorem 1.1), states that $k$ adaptive $(\varepsilon,\delta_0)$-DP mechanisms together satisfy

$$\left(\varepsilon\sqrt{2k\log(1/\eta)}+\frac{k\varepsilon(e^\varepsilon-1)}2,\;k\delta_0+\eta\right)\text{-DP}\qquad(0<\eta<1). \tag{3.5}$$

The individual mechanism's slack, often written $\delta'$, is renamed $\delta_0$ here, and the new parameter chosen when the theorem is applied, often written $\delta$, is renamed $\eta$. Both are additive terms in the final guarantee, but only $\eta$ is a free choice. The condition $\eta>0$ is needed for $\log(1/\eta)$ to be finite; at $\eta=0$, basic composition is still available.

## Where the improved mean bound comes from

For pure DP, $R=p/q$ satisfies $a=e^{-\varepsilon}\le R\le b=e^\varepsilon$ $Q$-almost surely, and $\mathbb E_Q[R]=\int p\,\mathrm d\nu=1$. The mean privacy loss is a Kullback–Leibler (KL) divergence:

$$\mathbb E_P[L]=\int p\log\frac pq\,\mathrm d\nu=\int qR\log R\,\mathrm d\nu=\mathbb E_Q[R\log R]=D_{\mathrm{KL}}(P\|Q).$$

Why consider $r\log r$? It is exactly the function inside the expectation we need to bound. It is convex, since $(r\log r)''=1/r>0$, so on $[a,b]$ its graph lies below the chord joining its values at $a$ and $b$:

$$R\log R\le\frac{b-R}{b-a}\,a\log a+\frac{R-a}{b-a}\,b\log b\qquad Q\text{-a.s.}$$

The right-hand side is linear in $R$, so taking $\mathbb E_Q$ (justified by the almost-sure inequality) simply replaces $R$ by $\mathbb E_Q[R]=1$:

$$D_{\mathrm{KL}}(P\|Q)\le\frac{b-1}{b-a}\,a\log a+\frac{1-a}{b-a}\,b\log b .$$

**The algebra.** Substitute $\log a=-\varepsilon$, $\log b=\varepsilon$ and $ab=1$. The numerator is $\varepsilon\bigl[-(b-1)a+(1-a)b\bigr]=\varepsilon(a+b-2ab)=\varepsilon(a+b-2)$. Writing $s=e^{\varepsilon/2}$, we have $a+b-2=(s-s^{-1})^2$ and $b-a=(s-s^{-1})(s+s^{-1})$, so

$$D_{\mathrm{KL}}(P\|Q)\le\varepsilon\,\frac{s-s^{-1}}{s+s^{-1}}=\varepsilon\tanh\frac\varepsilon2=\varepsilon\,\frac{e^\varepsilon-1}{e^\varepsilon+1}\le\frac{\varepsilon(e^\varepsilon-1)}2. \tag{3.6}$$

The last step uses $e^\varepsilon+1\ge2$. The looser right-hand side is the mean term in (3.5); the chord gives the sharper $\varepsilon\tanh(\varepsilon/2)$, which is at most $\varepsilon^2/2$ because $\tanh u\le u$ for $u\ge0$.

## Where the square-root term comes from {#sec-sqrt}

We prove (3.5) for $\delta_0=0$. Sample the transcript on dataset $x$, so all probabilities are under $P$. Let $\mathcal F_{i-1}$ be the $\sigma$-field of the first $i-1$ outputs. Conditionally on the history, round $i$ is an $\varepsilon$-DP mechanism, so its mean loss is a KL divergence bounded by (3.6):

$$\mu_i=\mathbb E[L_i\mid\mathcal F_{i-1}]\le\mu_0=\frac{\varepsilon(e^\varepsilon-1)}2 .$$

Let $X_i=L_i-\mu_i$. Then $\mathbb E[X_i\mid\mathcal F_{i-1}]=0$, and since $L_i\in[-\varepsilon,\varepsilon]$, $X_i$ lies in $[-\varepsilon-\mu_i,\varepsilon-\mu_i]$, an interval of width $2\varepsilon$ that is fixed once the history is. Hoeffding's lemma, applied conditionally, gives

$$\mathbb E[e^{\lambda X_i}\mid\mathcal F_{i-1}]\le e^{\lambda^2\varepsilon^2/2}.$$

**The tower step.** Let $S_j=\sum_{i\le j}X_i$. Because $S_{j-1}$ is fixed by $\mathcal F_{j-1}$,

$$\mathbb E[e^{\lambda S_j}]=\mathbb E\bigl[e^{\lambda S_{j-1}}\,\mathbb E[e^{\lambda X_j}\mid\mathcal F_{j-1}]\bigr]\le e^{\lambda^2\varepsilon^2/2}\,\mathbb E[e^{\lambda S_{j-1}}],$$

and by induction $\mathbb E[e^{\lambda S_k}]\le e^{k\lambda^2\varepsilon^2/2}$. Chernoff's bound (3.4) then gives, for every $\lambda>0$,

$$\Pr[S_k\ge t]\le\exp\left(-\lambda t+\frac{k\lambda^2\varepsilon^2}{2}\right).$$

The exponent is a parabola in $\lambda$ with its minimum where $-t+k\lambda\varepsilon^2=0$, that is at $\lambda=t/(k\varepsilon^2)$. There it equals $-t^2/(2k\varepsilon^2)$, so $\Pr[S_k\ge t]\le\exp[-t^2/(2k\varepsilon^2)]$. Choosing $t=\varepsilon\sqrt{2k\log(1/\eta)}$ makes this at most $\eta$.

Finally $L_{1:k}=S_k+\sum_i\mu_i\le S_k+k\mu_0$, so

$$\Pr_P\bigl[L_{1:k}>\varepsilon_*\bigr]\le\eta,\qquad\varepsilon_*=\varepsilon\sqrt{2k\log(1/\eta)}+k\mu_0 .$$

Lemma 3.1 turns this tail bound into $P(S)\le e^{\varepsilon_*}Q(S)+\eta$. The hypotheses are symmetric in $x$ and $x'$, so the same argument with the roles exchanged gives the other direction. This proves (3.5) when $\delta_0=0$.

**When $\delta_0>0$.** One needs an extra reduction, which we cite rather than prove. An $(\varepsilon,\delta_0)$-indistinguishable pair can be written as $P=(1-\delta_0)P'+\delta_0P''$ and $Q=(1-\delta_0)Q'+\delta_0Q''$, where $P'$ and $Q'$ are $\varepsilon$-indistinguishable (DRV10; Kairouz–Oh–Viswanath 2015 give the canonical form). Running the argument above on the primed parts and charging each round's exceptional part to the slack costs at most $k\delta_0$ in total. It is **not** valid to assume that $(\varepsilon,\delta_0)$-DP bounds $\Pr[L>\varepsilon]$ by $\delta_0$: the example in @sec-tail shows that the probability can be far larger.

# CDP: mean-concentrated differential privacy (Dwork–Rothblum) {#sec-cdp}

## Motivation and definition

Advanced composition used two facts about $L$ beyond its worst-case value: its mean, and how fast deviations above the mean become unlikely. Dwork–Rothblum make these two quantities the *definition* of privacy.

A real random variable $X$ is $\tau$-subgaussian if $\mathbb E[e^{\lambda(X-\mathbb EX)}]\le\exp(\lambda^2\tau^2/2)$ for every real $\lambda$. A mechanism is $(\mu,\tau)$-CDP when, for every ordered neighbouring pair, the privacy loss under $P$ satisfies

$$\mathbb E_P[L]\le\mu,\qquad\mathbb E_P\bigl[e^{\lambda(L-\mathbb E_PL)}\bigr]\le e^{\lambda^2\tau^2/2}\quad\text{for every }\lambda\in\mathbb R. \tag{4.1}$$

This version of CDP is also called mean-concentrated DP (mCDP), to distinguish it from zCDP. Unlike pure DP, it does not require a finite worst-case bound on the loss.

## CDP to approximate DP

Apply Chernoff's bound (3.4) to $X=L-\mathbb EL$ with the subgaussian bound:

$$\Pr[L-\mathbb EL\ge t]\le\inf_{\lambda>0}\exp\left(-\lambda t+\frac{\lambda^2\tau^2}2\right).$$

The exponent is minimized where $-t+\lambda\tau^2=0$, at $\lambda=t/\tau^2$, where it equals $-t^2/(2\tau^2)$:

$$\Pr[L-\mathbb EL\ge t]\le e^{-t^2/(2\tau^2)}.$$

Choose $t=\tau\sqrt{2\log(1/\eta)}$ to make the right-hand side $\eta$. Since $\mathbb EL\le\mu$, the event $\{L>\mu+t\}$ is contained in $\{L-\mathbb EL\ge t\}$, so

$$\Pr\bigl[L>\mu+\tau\sqrt{2\log(1/\eta)}\bigr]\le\eta .$$

Lemma 3.1, applied to both ordered pairs, gives for each $0<\eta<1$

$$(\mu,\tau)\text{-CDP}\Longrightarrow\bigl(\mu+\tau\sqrt{2\log(1/\eta)},\,\eta\bigr)\text{-DP}. \tag{4.2}$$

## Adaptive CDP composition: what the simple argument proves

Suppose that for every history, round $i$ is $(\mu_i,\tau_i)$-CDP. Let $m_i=\mathbb E[L_i\mid\mathcal F_{i-1}]$, the conditional mean, which is at most $\mu_i$ but depends on the history. For $\lambda>0$, replacing $m_i$ by the larger constant $\mu_i$ can only lower the exponent, so

$$\mathbb E\bigl[e^{\lambda(L_i-\mu_i)}\mid\mathcal F_{i-1}\bigr]\le\mathbb E\bigl[e^{\lambda(L_i-m_i)}\mid\mathcal F_{i-1}\bigr]\le e^{\lambda^2\tau_i^2/2}.$$

The tower step of @sec-sqrt then gives

$$\mathbb E\Bigl[e^{\lambda(L_{1:k}-\sum_i\mu_i)}\Bigr]\le\exp\Bigl(\tfrac{\lambda^2}{2}\textstyle\sum_i\tau_i^2\Bigr)\qquad(\lambda>0). \tag{4.3}$$

This is a **one-sided, uncentred** bound, and it is all that the conversion (4.2) uses: rerunning that derivation with (4.3) gives

$$k\text{ adaptive rounds}\Longrightarrow\Bigl(\textstyle\sum_i\mu_i+\sqrt{\sum_i\tau_i^2}\,\sqrt{2\log(1/\eta)},\;\eta\Bigr)\text{-DP}. \tag{4.4}$$

For identical rounds this is $k\mu+\tau\sqrt{2k\log(1/\eta)}$, the mean-plus-square-root structure of advanced composition.

**What this argument does not prove.** The statement that the composition is again $(\sum_i\mu_i,\sqrt{\sum_i\tau_i^2})$-CDP, in the sense of (4.1), does not follow from multiplying conditional MGFs. Definition (4.1) centres $L_{1:k}$ at its unconditional mean $\sum_i\mathbb E[m_i]$, but the conditional bounds centre each $L_i$ at the random $m_i$; the difference $\sum_i(m_i-\mathbb Em_i)$ is not controlled by the per-round hypotheses. For $\lambda<0$, moreover, replacing $m_i$ by $\mu_i$ moves the inequality the wrong way. Dwork–Rothblum (2016) state a composition theorem for CDP; Bun–Steinke (2016) discuss the technical difficulties of the mCDP definition. This note relies only on (4.3)–(4.4), which it proves.

## Pure DP and Gaussian noise as CDP examples

**Pure DP.** Under $\varepsilon$-DP, $L\in[-\varepsilon,\varepsilon]$ $P$-a.s., so $L-\mathbb EL$ lies in an interval of width $2\varepsilon$ and Hoeffding's lemma makes it $\varepsilon$-subgaussian. With the mean bound (3.6),

$$\varepsilon\text{-DP}\Longrightarrow\left(\frac{\varepsilon(e^\varepsilon-1)}2,\;\varepsilon\right)\text{-CDP}.$$

**Gaussian noise.** Take a scalar Gaussian mechanism $M(x)=f(x)+N(0,\sigma^2)$ with $|f(x)-f(x')|\le\Delta$, and let $d=f(x)-f(x')$. A sample under $P$ is $Y=f(x)+Z$ with $Z\sim N(0,\sigma^2)$, so $Y-f(x)=Z$ and $Y-f(x')=Z+d$. The normalizing constants cancel, and

$$L(Y)=\frac{(Y-f(x'))^2-(Y-f(x))^2}{2\sigma^2}=\frac{(Z+d)^2-Z^2}{2\sigma^2}=\frac{d^2}{2\sigma^2}+\frac{dZ}{\sigma^2}. \tag{4.5}$$

So under $P$ the loss is Gaussian with mean $d^2/(2\sigma^2)$ and variance $d^2\sigma^2/\sigma^4=d^2/\sigma^2$. A centred Gaussian with variance $v$ has MGF $\mathbb Ee^{\lambda X}=e^{\lambda^2v/2}$, so the centred loss is $(|d|/\sigma)$-subgaussian, with equality. Bounding $|d|\le\Delta$,

$$M\text{ is }\left(\frac{\Delta^2}{2\sigma^2},\;\frac{\Delta}{\sigma}\right)\text{-CDP}. \tag{4.6}$$

The loss in (4.5) is unbounded, since $Z$ is, so the Gaussian mechanism is **not** $\varepsilon$-DP for any finite $\varepsilon$. This is the case that concentrated notions were built for.

# zCDP: the Rényi-divergence formulation (Bun–Steinke) {#sec-zcdp}

## Why introduce Rényi divergence?

CDP controls the whole MGF of the centred privacy loss, and @sec-cdp showed that the centring is what makes composition awkward. Bun–Steinke instead bound the **Rényi divergence** at every order $\alpha>1$. This keeps the useful tail bounds and makes adaptive composition clean.

For $P\ll Q$, the Rényi divergence of order $\alpha>1$ is

$$D_\alpha(P\|Q)=\frac1{\alpha-1}\log\int\left(\frac{\mathrm dP}{\mathrm dQ}\right)^{\alpha}\mathrm dQ, \tag{5.1}$$

and $D_\alpha(P\|Q)=+\infty$ when $P\not\ll Q$. Using $R=e^L$, the integral is $\mathbb E_Q[e^{\alpha L}]$. Since $\mathrm dP=R\,\mathrm dQ$,

$$\mathbb E_Q[e^{\alpha L}]=\mathbb E_Q\bigl[R\cdot R^{\alpha-1}\bigr]=\mathbb E_P\bigl[R^{\alpha-1}\bigr]=\mathbb E_P\bigl[e^{(\alpha-1)L}\bigr],$$

and therefore

$$D_\alpha(P\|Q)=\frac1{\alpha-1}\log\mathbb E_P\bigl[e^{(\alpha-1)L}\bigr]. \tag{5.2}$$

This is the exact bridge between the privacy loss and Rényi divergence: $D_\alpha$ is the cumulant generating function of $L$ under $P$, evaluated at $\lambda=\alpha-1$ and divided by $\lambda$. In the limit $\alpha\downarrow1$, $D_\alpha\to D_{\mathrm{KL}}$ under the usual regularity assumptions, and as $\alpha\to\infty$, $D_\alpha$ tends to the max-divergence $\operatorname{ess\,sup}_PL$.

## Definition of zCDP and its MGF form

A mechanism is $(\xi,\rho)$-zCDP when, for all neighbouring ordered pairs and all $\alpha>1$,

$$D_\alpha(P\|Q)\le\xi+\rho\alpha,\qquad\xi,\rho\ge0. \tag{5.3}$$

When $\xi=0$ one simply says $\rho$-zCDP.

Insert (5.3) into (5.2) and set $\lambda=\alpha-1>0$, so that $\alpha=1+\lambda$:

$$\log\mathbb E_P[e^{\lambda L}]\le\lambda\bigl(\xi+\rho(1+\lambda)\bigr)=\lambda(\xi+\rho)+\rho\lambda^2,$$

or equivalently

$$\mathbb E_P\bigl[e^{\lambda(L-\xi-\rho)}\bigr]\le e^{\rho\lambda^2},\qquad\lambda>0. \tag{5.4}$$

Compared with a Gaussian MGF $e^{\lambda^2\tau^2/2}$, the variance proxy is $\tau^2=2\rho$ and the location is $\xi+\rho$. This is a one-sided, uncentred statement, the same shape as (4.3); it is not the mCDP definition, which bounds a centred MGF for every real $\lambda$.

## zCDP to approximate DP

For $t>0$ and $\lambda>0$, Chernoff's bound (3.4) with (5.4) gives

$$\Pr[L>\xi+\rho+t]\le\exp(-\lambda t+\rho\lambda^2).$$

For $\rho>0$ the exponent is minimized where $-t+2\rho\lambda=0$, at $\lambda=t/(2\rho)$. There it equals $-t^2/(2\rho)+t^2/(4\rho)=-t^2/(4\rho)$, so

$$\Pr[L>\xi+\rho+t]\le e^{-t^2/(4\rho)}.$$

Choose $t=2\sqrt{\rho\log(1/\eta)}$; then $t^2/(4\rho)=\log(1/\eta)$ and the tail probability is at most $\eta$. Lemma 3.1, in both directions, proves

$$(\xi,\rho)\text{-zCDP}\Longrightarrow\bigl(\xi+\rho+2\sqrt{\rho\log(1/\eta)},\;\eta\bigr)\text{-DP},\qquad0<\eta<1. \tag{5.5}$$

When $\rho=0$, the Rényi bound does not depend on $\alpha$; letting $\alpha\to\infty$ bounds the max-divergence by $\xi$, which is pure $\xi$-DP. Formula (5.5) remains a valid, weaker statement in that case.

The conversion (5.5) optimizes only $\lambda$ in a Chernoff bound, so it is not tight. Sharper conversions exist (Bun–Steinke 2016; Canonne–Kamath–Steinke 2020), and for the Gaussian mechanism the exact answer is computed in @sec-worked.

## Adaptive composition of zCDP

Define each $L_i$ from the conditional transcript densities, as in @sec-pure, and suppose every round is $(\xi_i,\rho_i)$-zCDP for every history. Fix $\alpha>1$. By (5.2) and (5.3), for every history,

$$\mathbb E\bigl[e^{(\alpha-1)L_i}\mid\mathcal F_{i-1}\bigr]\le\exp\bigl((\alpha-1)(\xi_i+\rho_i\alpha)\bigr).$$

The tower step of @sec-sqrt, with $\lambda=\alpha-1$ and the uncentred $L_i$ in place of $X_i$, bounds the MGF of $L_{1:k}=\sum_iL_i$ by the product of these bounds. Taking logarithms and dividing by $\alpha-1$ gives

$$D_\alpha(P_{1:k}\|Q_{1:k})\le\sum_i\xi_i+\alpha\sum_i\rho_i .$$

No centring appears, which is why the argument that failed for mCDP works here. Since $\alpha$ was arbitrary,

$$\text{adaptive composition is }\Bigl(\textstyle\sum_i\xi_i,\;\sum_i\rho_i\Bigr)\text{-zCDP}. \tag{5.6}$$

Combined with (5.5), $k$ mechanisms that are each $\rho$-zCDP give $\bigl(k\rho+2\sqrt{k\rho\log(1/\eta)},\,\eta\bigr)$-DP. Both zCDP parameters add linearly, while the final approximate-DP $\varepsilon$ has a square-root term.

## Post-processing

Rényi divergence satisfies the data-processing inequality: for any (possibly randomized) map $g$, $D_\alpha(g(P)\|g(Q))\le D_\alpha(P\|Q)$ (cited, not proved here; see van Erven–Harremoës 2014). Hence zCDP is closed under post-processing. Bun–Steinke (2016) point out that mCDP, as defined by (4.1), is not.

## The Gaussian mechanism has an exact zCDP parameter

Return to the Gaussian privacy loss (4.5), with $d=f(x)-f(x')$:

$$L\sim N\left(\frac{d^2}{2\sigma^2},\;\frac{d^2}{\sigma^2}\right)\quad\text{under }P.$$

For a Gaussian $N(m,v)$, $\log\mathbb Ee^{\lambda L}=\lambda m+\lambda^2v/2$, so here $\log\mathbb E_P[e^{\lambda L}]=\lambda d^2/(2\sigma^2)+\lambda^2d^2/(2\sigma^2)=\lambda(1+\lambda)\,d^2/(2\sigma^2)$. Divide by $\lambda=\alpha-1$ and use $1+\lambda=\alpha$:

$$D_\alpha(P\|Q)=\frac{\alpha d^2}{2\sigma^2}\le\alpha\,\frac{\Delta^2}{2\sigma^2}.$$

This holds with equality when $|d|=\Delta$, so the Gaussian mechanism satisfies

$$\boxed{\;\rho=\frac{\Delta^2}{2\sigma^2}\text{-zCDP},\;} \tag{5.7}$$

and no smaller $\rho$ works. This agrees with the CDP parameters (4.6), $\mu=\rho$ and $\tau=\sqrt{2\rho}$. For a vector query with independent noise in each coordinate, the same calculation holds with $\Delta$ the $\ell_2$-sensitivity.

## Pure DP implies zCDP

Under $\varepsilon$-DP, $L\in[-\varepsilon,\varepsilon]$ almost surely, and Hoeffding's lemma applied to $L-\mathbb E_PL$ gives

$$\log\mathbb E_P[e^{\lambda L}]\le\lambda\,\mathbb E_PL+\frac{\lambda^2\varepsilon^2}{2}.$$

The chord bound (3.6) gives $\mathbb E_PL\le\varepsilon\tanh(\varepsilon/2)\le\varepsilon^2/2$, so

$$\log\mathbb E_P[e^{\lambda L}]\le\frac{\varepsilon^2}{2}(\lambda+\lambda^2)=\frac{\varepsilon^2}{2}\,\lambda(1+\lambda).$$

Divide by $\lambda=\alpha-1$ and substitute $1+\lambda=\alpha$: $D_\alpha(P\|Q)\le\alpha\varepsilon^2/2$. Therefore

$$\varepsilon\text{-DP}\Longrightarrow\frac{\varepsilon^2}{2}\text{-zCDP}. \tag{5.8}$$

This route explains why a quadratic parameter appears when a pure-DP guarantee is converted to a concentrated one. It needs the sharper mean bound $\varepsilon\tanh(\varepsilon/2)$; the looser $\varepsilon(e^\varepsilon-1)/2$ exceeds $\varepsilon^2/2$ and would not give (5.8).

## From mCDP to zCDP

The two notions are related in one direction, which follows from (5.2). Under $(\mu,\tau)$-CDP, for $\lambda>0$,

$$\log\mathbb E_P[e^{\lambda L}]=\lambda\,\mathbb E_PL+\log\mathbb E_P\bigl[e^{\lambda(L-\mathbb E_PL)}\bigr]\le\lambda\mu+\frac{\lambda^2\tau^2}{2}.$$

Divide by $\lambda=\alpha-1$: $D_\alpha(P\|Q)\le\mu+(\alpha-1)\tau^2/2=(\mu-\tau^2/2)+\alpha\,\tau^2/2$. If $\mu<\tau^2/2$ the constant is negative and can be dropped. Hence

$$(\mu,\tau)\text{-CDP}\Longrightarrow\Bigl(\max\{0,\,\mu-\tfrac{\tau^2}2\},\;\tfrac{\tau^2}2\Bigr)\text{-zCDP}. \tag{5.9}$$

For the Gaussian mechanism, $\mu=\tau^2/2=\rho$, and (5.9) returns exactly (5.7). The converse does not follow from this note's arguments: zCDP controls $L$ only through (5.4), a one-sided bound for $\lambda>0$, while mCDP needs a centred bound for every real $\lambda$.

# Comparing the guarantees without confusing the symbols {#sec-compare}

| Notion | What it bounds | Adaptive composition | Conversion to $(\varepsilon,\eta)$-DP | Post-processing |
|:---|:---|:---|:---|:---|
| Pure $\varepsilon$-DP | Every event-probability ratio; $\lvert L\rvert\le\varepsilon$ a.s. | $\sum_i\varepsilon_i$, pure DP | is already DP | closed |
| $(\varepsilon,\delta)$-DP | $H_\varepsilon\le\delta$ in both directions | basic: $(\sum\varepsilon_i,\sum\delta_i)$; advanced: (3.5) | is already DP | closed |
| $(\mu,\tau)$-CDP (mCDP) | Mean loss $\le\mu$; centred MGF $\le e^{\lambda^2\tau^2/2}$ | one-sided bound (4.3), giving (4.4) | (4.2) | not closed (BS16) |
| $(\xi,\rho)$-zCDP | Every Rényi divergence $D_\alpha\le\xi+\rho\alpha$ | $(\sum\xi_i,\sum\rho_i)$-zCDP, (5.6) | (5.5) | closed |

Bridges derived above: $\varepsilon$-DP $\Rightarrow$ $(\varepsilon(e^\varepsilon-1)/2,\varepsilon)$-CDP; $\varepsilon$-DP $\Rightarrow$ $\varepsilon^2/2$-zCDP (5.8); $(\mu,\tau)$-CDP $\Rightarrow$ zCDP (5.9); Gaussian: $\rho=\Delta^2/(2\sigma^2)$, $\mu=\rho$, $\tau=\sqrt{2\rho}$.

**Parameter warning.** Sources use $\delta$ both for the additive slack of an approximate-DP guarantee **and** for a failure probability chosen freely during a conversion or in advanced composition. The two play different roles in the derivations but end up added together in an approximate-DP statement. This note uses $\eta$ for a newly chosen tail or conversion parameter, and keeps $\delta$ or $\delta_0$ for the slack of an existing mechanism's guarantee.

# Worked comparison: repeating a Gaussian query {#sec-worked}

A scalar query has sensitivity $\Delta=1$ and is answered with Gaussian noise of standard deviation $\sigma=5$, $k=100$ times (adaptively if you like, provided every round has sensitivity at most $1$ given the history). The target is $\delta_{\mathrm{total}}=10^{-5}$.

**Through zCDP.** Each round has $\rho=1/(2\cdot25)=0.02$. Composition (5.6) gives $\rho_{\mathrm{total}}=100\times0.02=2$. With $\eta=10^{-5}$, so that $\log(1/\eta)=11.513$, the conversion (5.5) gives

$$\varepsilon_{\mathrm{total}}=2+2\sqrt{2\times11.513}=2+9.597\approx11.60,\qquad\delta_{\mathrm{total}}=10^{-5}.$$

**Through mCDP.** Each round is $(0.02,\,0.2)$-CDP by (4.6), so (4.4) with $\sum\mu_i=2$ and $\sqrt{\sum\tau_i^2}=\sqrt{100\times0.04}=2$ gives $2+2\sqrt{2\times11.513}\approx11.60$, the same number. That is no accident: for Gaussian noise the two definitions carry the same information.

**The exact answer.** For a pair of Gaussians, (3.3) can be evaluated in closed form. Write $\mu_G=|d|/\sigma$. By (4.5), under $P$ the loss is $L=\mu_G^2/2+\mu_GW$ with $W\sim N(0,1)$. Under $Q$ the same calculation with $Y=f(x')+Z$ gives $L=-\mu_G^2/2+\mu_GW$. The optimal event in (3.2) is $\{L>\varepsilon\}$, so $H_\varepsilon=P(L>\varepsilon)-e^\varepsilon Q(L>\varepsilon)$, that is,

$$\delta(\varepsilon)=\Phi\!\left(\frac{\mu_G}2-\frac{\varepsilon}{\mu_G}\right)-e^{\varepsilon}\,\Phi\!\left(-\frac{\mu_G}2-\frac{\varepsilon}{\mu_G}\right), \tag{7.1}$$

where $\Phi$ is the standard normal distribution function (Balle–Wang 2018). One round has $\mu_G=\Delta/\sigma=0.2$. For $100$ non-adaptive rounds with $|d|=\Delta$, the total loss is a sum of $100$ independent copies of (4.5), which is again Gaussian, of mean $100\times0.02=2$ and variance $100\times0.04=4$. That is the loss of a single Gaussian with $\mu_G=\sqrt{k}\,\Delta/\sigma=2$. (That adaptive rounds are no worse is the composition theorem for Gaussian DP, Dong–Roth–Su 2022; cited, not derived.) Solving (7.1) for $\delta=10^{-5}$ with $\mu_G=2$ gives $\varepsilon\approx10.00$.

**Through per-round $(\varepsilon,\delta)$-DP.** Formula (7.1) with $\mu_G=0.2$ gives each round's $\varepsilon_0$ for a chosen $\delta_0$:

| Route | Per-round guarantee | $\varepsilon_{\mathrm{total}}$ at $\delta_{\mathrm{total}}=10^{-5}$ |
|:---|:---|---:|
| Basic composition | $(0.932,\;10^{-7})$-DP | $93.2$ |
| Advanced composition (3.5), best split of $\delta$ | $(0.938,\;8.5\times10^{-8})$-DP, $\eta=1.5\times10^{-6}$ | $121.6$ |
| zCDP or mCDP, then convert | $\rho=0.02$ | $11.60$ |
| Exact, (7.1) | $\mu_G=0.2$ | $10.00$ |

Advanced composition loses to basic composition here because $\varepsilon_0\approx0.94$ is not small: its mean term $k\varepsilon_0(e^{\varepsilon_0}-1)/2\approx72.8$ alone is most of the basic total. The concentrated notions avoid the problem because they never replace the Gaussian loss by a worst case.

All of these are **guarantees**, upper bounds that hold for every pair of neighbours; none says the realized privacy loss equals 11.60. Increasing the noise lowers $\rho$ quadratically: doubling $\sigma$ divides each $\rho$ by four. And never equate an $\varepsilon$-DP parameter numerically with a zCDP $\rho$: they measure different objects.

# Limitations and exact scope of the derivations

The steps for pure DP, the hockey-stick formula and the tail lemma, the KL chord bound, the subgaussian tails, the Rényi–MGF bridge, the Gaussian privacy loss, zCDP composition, and the CDP-to-zCDP bridge are derived in this note. Four results are cited rather than proved: Hoeffding's lemma; the decomposition of an $(\varepsilon,\delta_0)$-indistinguishable pair needed for advanced composition with $\delta_0>0$ (which is **not** implied by the false shortcut $\Pr[L>\varepsilon]\le\delta$); the data-processing inequality for Rényi divergence; and the adaptive composition theorem for Gaussian DP used for the exact figure. For mCDP, this note proves only the one-sided composition bound (4.3)–(4.4), not closure of the definition (4.1) under composition.

CDP/mCDP and zCDP are related, by (5.9), but not interchangeable. An arbitrary approximate-DP mechanism need not have finite Rényi divergence at any order, and hence need not satisfy zCDP: any mechanism with $\delta>0$ that puts mass where its neighbour puts none has $D_\alpha=\infty$. All comparisons assume one fixed adjacency convention (add/remove versus replace-one) and one privacy unit; sensitivities, and with them every number in @sec-worked, change with the convention.

# Primary sources and reading order

1. C. Dwork, G. Rothblum, S. Vadhan (2010), *Boosting and Differential Privacy*. Original advanced-composition method and the $O(\varepsilon^2)$ mean-loss result. <https://privacytools.seas.harvard.edu/sites/g/files/omnuum6656/files/privacytools/files/05670947.pdf>
2. C. Dwork, G. Rothblum (2016), *Concentrated Differential Privacy*. Theorem 1.1, the original $(\mu,\tau)$-CDP, concentration and Gaussian analysis, improved expected-loss bound. <https://arxiv.org/abs/1603.01887>
3. M. Bun, T. Steinke (2016), *Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds*. Definition of $(\xi,\rho)$-zCDP, Rényi-divergence conversions, additive composition, the Gaussian mechanism, and the relation to mCDP. <https://arxiv.org/abs/1605.02065>
4. P. Kairouz, S. Oh, P. Viswanath (2015), *The Composition Theorem for Differential Privacy*. Optimal composition beyond the standard advanced bound, and the canonical form of an $(\varepsilon,\delta)$ pair. <https://proceedings.mlr.press/v37/kairouz15.pdf>
5. B. Balle, Y.-X. Wang (2018), *Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising*. The exact privacy profile (7.1). <https://arxiv.org/abs/1805.06530>
6. J. Dong, A. Roth, W. Su (2022), *Gaussian Differential Privacy*. Composition of Gaussian mechanisms under adaptivity. <https://arxiv.org/abs/1905.02383>
7. C. Canonne, G. Kamath, T. Steinke (2020), *The Discrete Gaussian for Differential Privacy*. A sharper zCDP-to-DP conversion than (5.5). <https://arxiv.org/abs/2004.00010>
8. T. van Erven, P. Harremoës (2014), *Rényi Divergence and Kullback–Leibler Divergence*. Properties of $D_\alpha$, including data processing. <https://arxiv.org/abs/1206.2459>
