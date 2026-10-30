---
title: "Datasets for Differentially Private Machine Learning"
subtitle: "Which public data DP papers use, what each dataset can test, and where to get it"
abstract: |
  Every experiment with differential privacy (DP) runs on some dataset, and the choice decides what the experiment can show. This note catalogues the public datasets that DP papers use, by kind of data: tables, images, text, data with user identifiers, locations, graphs and clinical records. For each it gives the size, the licence, what one record is, and the results it is known for. One section covers the datasets and benchmarks of DP synthetic data. The note starts with what a dataset's size and shape mean for DP, with numbers from a script, and ends with how to choose a dataset for a study of the privacy-utility trade-off.
---

Suppose you want to run, or read, an experiment with differential privacy (DP). You need a dataset, and three things about it decide what the experiment can show. What one record is decides whom the guarantee protects. How many records there are decides whether the noise is small. And where the data came from decides whether you may use them and whether a model has already seen them. This note is a guide to the public datasets of DP papers, organized around those three questions.

The note assumes basic probability and what DP promises, and nothing about any dataset. Every size, licence and result below comes from the dataset's own page or from the paper that reports it, and the sources are listed at the end. Where the note could not confirm a number, it says so. @tbl-map gives the question each section answers, and @tbl-symbols lists the few symbols the note uses.

| Section | The question it answers |
|:---|:---|
| @sec-what | What do the size and shape of a dataset mean for DP? |
| @sec-tabular | Which tables do DP papers use, and what has replaced Adult? |
| @sec-images | Which image datasets, and what accuracy does DP reach on them? |
| @sec-text | Which text datasets, and what does public pre-training change? |
| @sec-users | Which datasets have user identifiers, and what does user-level privacy cost? |
| @sec-other | What is used for locations, graphs and clinical records? |
| @sec-released | Which real DP releases are themselves public datasets? |
| @sec-synthetic | What do DP synthetic data generators train and compete on? |
| @sec-choose | How do you choose a dataset for a privacy-utility study? |

: The question each section answers. {#tbl-map}

| Symbol | Meaning |
|:---|:---|
| $D$, $D'$; $D \sim D'$ | two neighbouring datasets, which differ by one record added or removed |
| $M$ | a mechanism, the random procedure that computes a released output |
| $\varepsilon$, $\delta$ | the parameters of a DP guarantee |
| $n$ | the number of records in a dataset |
| $s$ | the share of the records that a count covers |
| $k$ | the most records one user may contribute, the contribution bound |
| $\rho$ | the single parameter of zero-concentrated DP, a form of the guarantee that converts to an $(\varepsilon, \delta)$ pair |

: The symbols the note uses. Each is also explained where it first appears. {#tbl-symbols}

# What a dataset has to offer a DP experiment {#sec-what}

Four questions come up for every dataset in the catalogue, so this section answers them once. What is one record? Are the records enough? Is the range of each value known? And what does the licence allow? All the numbers of the note come from one script, [code/notes/dp_datasets.py](https://github.com/manuelvloureiro/privacy-preserving-machine-learning-book/blob/main/code/notes/dp_datasets.py).

## The privacy unit

DP protects whatever one record is, and its definition shows how. Write $M$ for a mechanism, the random procedure that computes the released output, and take two datasets $D$ and $D'$ that differ in one record. The privacy parameter $\varepsilon$ bounds how much the probability of any output may change between the two datasets, and the parameter $\delta$ is a small probability of exception. The mechanism is $(\varepsilon, \delta)$-DP when, for any such pair of datasets and any set $S$ of outputs,

$$\Pr[M(D) \in S] \le e^{\varepsilon}\Pr[M(D') \in S] + \delta.$$

This note uses the add/remove relation, so $D'$ is $D$ with one record added or taken away. The guarantee is thus a promise about one record, and the dataset decides what a record is. In Adult one record is one person. In Online Retail one record is one line of an invoice, so a customer with 400 lines is 400 records, and a guarantee per record says little about the customer. To protect the customer, the mechanism must treat all of their lines as one unit, which is **user-level** privacy, and the dataset must say which lines belong to whom. Most benchmarks have no such identifier, so they can only test **example-level** privacy, and @sec-users is about the ones that do.

## The size of the dataset

The noise that DP adds to a count does not depend on the number of records. Under add/remove, one record changes a count by at most one, so Laplace noise with scale $1/\varepsilon$ makes the count $\varepsilon$-DP, and that noise has standard deviation $\sqrt{2}/\varepsilon$ for any dataset. The number of records $n$ decides how this fixed noise compares with the counts. A count that covers a share $s$ of the records is about $ns$, so the noise, as a fraction of the count, is

$$\frac{\sqrt{2}}{\varepsilon\, n s}.$$

So twice the data halves the relative error, and a ten times smaller $\varepsilon$ costs as much as ten times less data.

@fig-noise-floor plots that fraction against $n$ for a count of 1% of the records, with the common benchmarks marked on the $\varepsilon = 1$ line. The dotted line is a relative error of 10%, a reasonable bar for a statistic worth publishing. With $s = 0.01$, the fraction falls to 0.1 when $n = \sqrt{2}/(0.001\,\varepsilon)$. So a dataset clears the bar with 1,414 records at $\varepsilon = 1$, and with 14,142 at $\varepsilon = 0.1$. @tbl-sizes gives the values for each dataset.

![The Laplace noise on a count of 1% of the dataset, as a share of that count, against the number of records. Each marker is a benchmark dataset at ε = 1.](figures/noise_floor.svg){#fig-noise-floor}

{{< include figures/_sizes.md >}}

Two things follow for the choice of a dataset. First, the small datasets of machine-learning courses, with a few hundred or a thousand records, cannot show DP at work, because the noise swamps any count finer than the whole. For example, a 1% count of German Credit's 1,000 applicants is ten people, and the noise at $\varepsilon = 1$ is 14% of it. Second, above about fifty thousand records a single count is almost free at any $\varepsilon$ a paper would report. On such data the cost of DP comes from many questions, because they share the budget, or from the training of a model, which asks thousands.

## The domain

A DP mechanism must know the range of every value before it looks at the data. The reason is that the sensitivity of a sum is the largest value one record can add, and a largest value read off the data is itself a leak. So a dataset with a public codebook, which lists each attribute's categories and bounds, is easier to use than one whose columns you must inspect. Adult's codebook gives every category and the survey's top codes, the caps above which the survey records a value as the cap itself. The size of the domain also decides what a mechanism can release, and @sec-synthetic takes this up.

## Licence and provenance

A paper downloads a dataset, changes it and sometimes shares it again, and each of these needs permission. The datasets of the UCI Machine Learning Repository carry a Creative Commons licence (CC BY 4.0), which allows all three with attribution. Kaggle datasets carry whatever licence the uploader chose, ImageNet and CelebA are for non-commercial research only, and clinical data such as MIMIC need a signed agreement and forbid you to share them. The catalogue names the terms of each entry, because the choice often depends on them.

Provenance matters for a different reason, because benchmark datasets are not private. Adult is a public survey extract, and CIFAR-10 was collected from the web. So a DP experiment on them measures what a mechanism would cost on data of that shape, not whether anyone is harmed, which is the right measurement. But public data can also make a result look better than it is. A model pre-trained on a public web corpus may already have seen the benchmark on which it is then fine-tuned with DP, and its DP accuracy is then too high. @sec-text returns to this.

# Tabular data {#sec-tabular}

Tables are where DP started, because a statistics office releases tables of survey records. Most DP papers on query release, which answers many counting queries about a table under one budget, still use tables, and so do most papers on synthetic data. @tbl-tabular lists the common ones, from the dataset that nearly every paper uses to the census microdata that is replacing it.

| Dataset | Records | One record is | Label or task | Licence |
|:---|---:|:---|:---|:---|
| Adult (UCI 2) | 48,842 | a person, 1994 survey | income over $50,000 | CC BY 4.0 |
| ACSIncome (folktables, 2018, US) | 1,664,500 | a person, ACS PUMS | income over $50,000 | Census Bureau terms |
| ACSEmployment (folktables, 2018, US) | 3,236,107 | a person | employed | Census Bureau terms |
| Census-Income KDD (UCI 117) | 299,285 | a person, 1994–95 surveys | income over $50,000 | CC BY 4.0 |
| Bank Marketing (UCI 222) | 45,211 | a client contact | subscribed a deposit | CC BY 4.0 |
| German Credit (UCI 144) | 1,000 | a loan applicant | good or bad risk | CC BY 4.0 |
| Credit Card Fraud (Kaggle, ULB) | 284,807 | a transaction | fraud, 492 cases | DbCL 1.0 |
| Online Retail (UCI 352) | 541,909 | an invoice line | none | CC BY 4.0 |
| Diabetes 130-US hospitals (UCI 296) | 101,766 | a hospital stay | readmission | CC BY 4.0 |
| MovieLens 25M (GroupLens) | 25,000,095 | a rating | rating prediction | research use, no redistribution |

: The tabular datasets of this section. Record counts are as the source page states them. {#tbl-tabular}

## Adult, and why it is retiring

**Adult** is the table of more DP papers than any other. Barry Becker extracted it from the 1994 Current Population Survey ([UCI dataset 2](https://archive.ics.uci.edu/dataset/2/adult), licence CC BY 4.0). It holds 48,842 people with 14 attributes and a label that says whether their income was over $50,000. Its attributes are mostly categorical, its codebook lists every category, and it is small enough to run anything on. So it appears in query-release papers such as those of Hardt, Ligett and McSherry (2012) and Aydore et al. (2021), in every synthetic data benchmark of @sec-synthetic, and in the fairness literature. Some papers use a cleaned version of 45,222 rows without the missing values, so a paper's row count tells you which version it took.

Adult's weaknesses are the reason it is retiring: its data are from 1994, its $50,000 threshold is arbitrary, and nobody documented how the survey was filtered. Ding, Hardt, Miller and Schmidt (2021) built **folktables** to replace it. The library downloads the American Community Survey (ACS) Public Use Microdata Sample (PUMS) from the Census Bureau and derives five prediction tasks from it, for any state and any year from 2014. ACSIncome is the direct successor of Adult, with 1,664,500 people and ten features for the whole United States in 2018. The code has the MIT licence, and the Census Bureau's terms of service allow use and redistribution of the data with attribution ([folktables](https://github.com/socialfoundations/folktables)).

For DP, folktables has two more advantages. First, you can cut a task by state, which gives datasets with the same columns from a few thousand records to millions, so a paper can show how the cost of privacy falls with $n$. Second, the same survey is the source of the Census Bureau's own DP work and of the NIST synthetic data benchmarks, so results carry over. For example, Vietri et al. (2022) built a synthetic data benchmark from five tasks in five states. A paper that once would have used Adult should now use ACSIncome for one state, and use Adult only to compare with older work.

## The other common tables

The **UCI Census-Income (KDD)** dataset is a larger relative of Adult, with 299,285 records and 41 attributes ([UCI dataset 117](https://archive.ics.uci.edu/dataset/117/census+income+kdd), CC BY 4.0). But use it with care, because its codebook reports tens of thousands of duplicate or conflicting rows. **Bank Marketing** records 45,211 telephone contacts of a Portuguese bank's marketing campaign, each with the client's attributes and whether they subscribed a term deposit ([UCI dataset 222](https://archive.ics.uci.edu/dataset/222/bank+marketing), CC BY 4.0). So it is a real campaign with a yes-or-no outcome. It exists in two versions with different columns, and papers rarely say which one they used.

**German Credit** holds 1,000 loan applicants with a good-or-bad label ([UCI dataset 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), CC BY 4.0). It is too small for DP, as @sec-what showed. The **Credit Card Fraud** dataset of the Université Libre de Bruxelles is the opposite case ([Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), Database Contents License 1.0). It holds 284,807 card transactions, and only 492 of them are frauds. Its features are 28 anonymized principal components, the time and the amount. The extreme imbalance makes it useful for DP, because noise can erase a class of 492 records. Two generators of @sec-synthetic use it. One is DP-MERF. The other is a generative adversarial network (GAN) trained through private aggregation of teacher ensembles (PATE), called PATE-GAN.

**Diabetes 130-US hospitals** is the largest clinical table that you can download freely, with 101,766 hospital stays and a readmission label ([UCI dataset 296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), CC BY 4.0). **Online Retail** records 541,909 invoice lines of a UK online shop over thirteen months ([UCI dataset 352](https://archive.ics.uci.edu/dataset/352/online+retail), CC BY 4.0). It has no label, and this note found no DP paper that uses it. But 406,829 of its lines have a customer id, which makes it the one table with a free licence on which you can study user-level privacy, and @sec-users does so.

Two Kaggle datasets are also in use. **Home Credit Default Risk** has 307,511 loan applications, and the **Lending Club** loan files are the source of the "Loans" dataset of Vietri et al. (2020) and Aydore et al. (2021). This note could not verify the terms of either, so a paper that uses them must say what the terms allow.

## Recommendation data, and the warning of Netflix

**MovieLens** is the standard dataset for recommendation, released by GroupLens for research on the condition that nobody redistributes it ([GroupLens](https://files.grouplens.org/datasets/movielens/)). Each record is one rating, and each user has many, so it is a user-level dataset. For example, Jain, Thakkar and Thakurta (2018) train DP matrix completion on it and on the **Netflix Prize** data, 100,480,507 ratings by 480,189 subscribers.

The Netflix data are a warning. Netflix released them in 2006 without the subscriber names, and Narayanan and Shmatikov (2008) then used public reviews on IMDb to show that eight ratings with approximate dates identify 99% of subscribers uniquely. Netflix withdrew the dataset after a lawsuit. On the same data, McSherry and Mironov (2009) showed that DP recommenders lose little accuracy. So "anonymized" means nothing without a definition, and DP is such a definition.

## The Census Bureau's own data

The US Census Bureau protected the 2020 Census with DP, and the data around that decision are a resource of their own. The TopDown algorithm of Abowd et al. (2022) adds discrete Gaussian noise to counts at every level of geography and then fits the published tables to those noisy measurements. The redistricting data have a total budget of $\varepsilon = 19.61$ ([Census Bureau, June 2021](https://www.census.gov/newsroom/press-releases/2021/2020-census-key-parameters.html)). So that researchers could study the effect, the Bureau also ran the algorithm on the 2010 Census and published the results as demonstration data ([2020 DAS development](https://www.census.gov/programs-surveys/decennial-census/decade/2020/planning-management/process/disclosure-avoidance/2020-das-development.html)). Kenny et al. (2021) used them to measure what DP did to redistricting. It published the noisy measurement file too, the counts after the noise and before the fit, which is the closest to a raw DP release that any statistics office has gone ([AWS Open Data](https://registry.opendata.aws/census-2010-pl94-nmf)).

The Bureau tested the algorithm on the full-count 1940 Census, which IPUMS distributes to registered users, and its Colorado extract became the data of the first NIST challenge in @sec-synthetic ([IPUMS](https://usa.ipums.org/usa/1940CensusDASTestData.shtml)). The ACS PUMS, the Bureau's other public microdata, is the source of folktables and of the NIST excerpts ([ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html)).

# Images {#sec-images}

A few image datasets carry most of the DP deep-learning literature, because the private form of stochastic gradient descent (SGD), called DP-SGD, started with image classification. DP-SGD clips the gradient of each example in a random batch and adds Gaussian noise to every update, and an accountant turns the noise and the number of steps into an $(\varepsilon, \delta)$ guarantee. @tbl-images lists the datasets with their best-known results.

| Dataset | Images | Classes | Licence or terms | A DP result |
|:---|---:|---:|:---|:---|
| MNIST | 60,000 + 10,000 | 10 | none stated | 98.7% at $\varepsilon = 2.93$ (Tramèr and Boneh, 2021) |
| Fashion-MNIST | 60,000 + 10,000 | 10 | MIT | 89.7% at $\varepsilon = 3$ (Tramèr and Boneh, 2021) |
| CIFAR-10 | 50,000 + 10,000 | 10 | none stated | 81.4% at $\varepsilon = 8$ from scratch (De et al., 2022) |
| CIFAR-100 | 50,000 + 10,000 | 100 | none stated | 81.8% at $\varepsilon = 8$ with ImageNet pre-training (De et al., 2022) |
| ImageNet (ILSVRC 2012) | 1,281,167 + 50,000 | 1,000 | non-commercial research | 32.4% at $\varepsilon = 8$ from scratch (De et al., 2022) |
| Places365-Standard | 1,803,460 | 365 | non-commercial, no redistribution | public pre-training data (Kurakin et al., 2022) |
| CelebA | 202,599 | 40 attributes | non-commercial research | FID 21.1 at $\varepsilon = 10$ (Dockhorn et al., 2022) |
| Camelyon17 (WILDS) | 455,954 patches | 2 | CC0 | 91.1% at $\varepsilon = 10$ (Ghalebikesabi et al., 2023) |

: The image datasets of DP deep learning. Train + test sizes are as the source page states them. Each result is at δ = 10⁻⁵ unless the paper states otherwise. {#tbl-images}

**MNIST** and **CIFAR-10** are the two datasets of Abadi et al. (2016), the paper that introduced DP-SGD, so every later paper reports on them to be comparable. MNIST's handwritten digits are so easy that DP costs little: the 98.7% of Tramèr and Boneh (2021) in @tbl-images is close to the non-private 99%. **Fashion-MNIST**, a drop-in replacement with clothing items, is harder ([Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)).

CIFAR-10 is where the cost of DP shows. Without privacy, models classify its colour images at over 95% ([CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)). With DP at $\varepsilon = 8$, De et al. (2022) reached 81.4% from scratch, and 96.7% with a model pre-trained on ImageNet. Read the second number with care, because it depends on what counts as public data, and @sec-text comes back to that.

**ImageNet** tests whether DP training scales ([ImageNet](https://www.image-net.org/download.php)). Kurakin et al. (2022) pre-trained a ResNet-18 on Places365 and then trained it with DP to 47.9% at $\varepsilon = 10$, against 75% without privacy. De et al. (2022) reached 32.4% from scratch at $\varepsilon = 8$, but 86.7% when they fine-tuned a model pre-trained on JFT-4B, a Google-internal dataset of four billion images. These two numbers for one dataset at one budget show that a DP result means nothing until its public data are stated.

**CelebA** and **Camelyon17** are the datasets of DP image synthesis, which @sec-synthetic covers ([CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)). For CelebA's faces, @tbl-images gives the Fréchet inception distance (FID) of DP diffusion models, a measure of how far generated images are from real ones. Camelyon17 is different in kind. It has histopathology patches from five Dutch hospitals, labelled for tumour tissue, in the public domain ([WILDS](https://wilds.stanford.edu/datasets/)). So it is the one image dataset in this note whose contents resemble data that would actually be private.

Two recent benchmark papers question these choices. Mokhtari et al. (2026) argue that nothing in MNIST, CIFAR-10 or ImageNet is private, and propose the medical datasets CheXpert and EyePACS for private fine-tuning. DPMLBench, by Wei et al. (2023), compares twelve DP training algorithms on four image datasets and finds that the ranking depends on the dataset.

# Text {#sec-text}

Language models memorize their training data, so DP matters for them. This section lists the datasets of DP language modelling, and then asks the question on which every DP text result now depends: what did the model see before DP training began?

| Dataset | Size | Task | Licence | A DP result |
|:---|---:|:---|:---|:---|
| SST-2 (GLUE) | 67,349 sentences | sentiment | not stated | 93.0% at $\varepsilon = 3$ (Li et al., 2022) |
| MNLI (GLUE) | 392,702 pairs | entailment | not stated | 87.8% at $\varepsilon = 6.7$ (Yu et al., 2022) |
| QNLI, QQP (GLUE) | 104,743 and 363,846 | question answering and paraphrase | not stated | 90.8% and 86.4% at $\varepsilon = 3$ (Li et al., 2022) |
| E2E NLG | 42,061 pairs | table to text | CC BY-SA 4.0 | BLEU 61.5 at $\varepsilon = 3$ (Li et al., 2022) |
| DART | 30,526 instances | table to text | MIT | BLEU 43.8 at $\varepsilon = 6.8$ (Yu et al., 2022) |
| PersonaChat | 162,064 utterances | dialogue | not stated | used by Li et al. (2022) |
| WikiText-103 | 103 million tokens | language modelling | CC BY-SA | not in the four papers checked |
| Enron emails | about 500,000 messages, 150 users | memorization studies | none | real personal mail |
| Yelp reviews (processed) | 1.9 million reviews | synthetic text | Yelp terms | see @sec-synthetic |

: The text datasets of DP fine-tuning. The GLUE sizes are training sets, and each result is development-set accuracy. {#tbl-text}

The DP fine-tuning papers of 2022 fixed the benchmarks. Li et al. (2022) and Yu et al. (2022) both fine-tune RoBERTa on four tasks of the GLUE benchmark of language understanding, and GPT-2 on E2E and DART ([GLUE sizes](https://www.tensorflow.org/datasets/catalog/glue)). @tbl-text gives their results. At $\varepsilon = 3$ a large model fine-tuned with DP is within a few points of its non-private accuracy on every GLUE task. On E2E the cost shows in the BLEU score, a measure from 0 to 100 of how closely generated text matches reference text, which falls from 69.5 to 61.5. GLUE's test labels are hidden, so papers report on the development sets.

The older language-modelling benchmarks, such as **WikiText-103** ([WikiText](https://huggingface.co/datasets/Salesforce/wikitext)), appear in none of the four papers above. The **Enron emails** are about half a million messages of 150 Enron employees, released by a court ([CMU](https://www.cs.cmu.edu/~enron/)). They are the standard testbed for memorization and extraction attacks, because they are real personal mail with real names. They have no licence, only a request to respect the privacy of the people in them, which is itself a reason to think before you use them.

## What the model saw first

Every result in @tbl-text comes from a model pre-trained on a public corpus and then fine-tuned with DP, and most of the accuracy comes from the pre-training. The corpora are Wikipedia and books for BERT and RoBERTa, WebText for GPT-2, and C4 or the Pile for later models. Anil et al. (2022) showed what DP costs without that help. They pre-trained BERT itself with DP on Wikipedia and books, 346 million examples. A masked-language-model accuracy of 60.5% at $\varepsilon = 5.36$, against about 70% without privacy, took a batch of two million examples.

Tramèr, Kamath and Carlini (2024) ask whether this recipe is honest. The first problem is that a web corpus contains personal data of people who never consented, so a model "pre-trained publicly" has already absorbed private information, and the DP guarantee on the fine-tuning says nothing about it. The second is that the corpus may also contain the benchmark itself. For example, C4 is 364,868,892 documents of filtered Common Crawl ([C4](https://huggingface.co/datasets/allenai/c4)), and the Pile is 825 GiB from 22 sources ([the Pile](https://pile.eleuther.ai/)). So a paper with a DP result on text should state the pre-training corpus, and a reader should treat the public data as part of the method. Ponomareva et al. (2023) put this in their reporting checklist, next to the privacy unit and the neighbouring relation.

# Datasets with user identifiers {#sec-users}

A guarantee about a person needs a dataset that knows which records belong to the person. Few benchmarks do, and most of those come from federated learning. This section lists them in @tbl-users and then uses Online Retail to show what user-level privacy costs.

| Dataset | Users | Records | Records per user | Licence |
|:---|---:|---:|---:|:---|
| Stack Overflow (TFF) | 342,477 train | 135,818,730 | about 400 | CC BY-SA 3.0 |
| Reddit (LEAF) | 1,660,820 | 56,587,343 | 34 | not stated |
| FEMNIST (LEAF) | 3,550 | 805,263 | 227 | not stated |
| Shakespeare (TFF) | 715 | 16,068 | 22 | public-domain text |
| CelebA (LEAF) | 9,343 | 200,288 | 21 | non-commercial research |
| Sentiment140 (LEAF) | 660,120 | 1,600,498 | 2.4 | research use |
| FLAIR (Apple) | 51,414 | 429,078 | 8.3 | CC BY-NC 4.0 annotations |
| Online Retail (UCI) | 4,372 | 406,829 | 93 (median 42) | CC BY 4.0 |
| MovieLens 25M | 162,541 | 25,000,095 | 154 | research use |

: The datasets with user identifiers. Records per user is the mean as the source reports it, or as the script computes it for Online Retail. {#tbl-users}

Most of these datasets come from the **LEAF** benchmark of Caldas et al. (2018) and the datasets module of TensorFlow Federated (TFF). **Stack Overflow** is the largest, with 342,477 training users ([TFF](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data)). Kairouz et al. (2021) used it to test the private form of follow the regularized leader (FTRL), called DP-FTRL. This method adds noise that is correlated across steps, so it does not need DP-SGD's random batches, which a deployment on phones cannot guarantee. With user-level privacy at $\varepsilon \approx 18.7$ it trains a next-word model to about 21% accuracy, against 25% without privacy. On comments from **Reddit**, McMahan et al. (2018) trained the first user-level DP language model and lost almost no accuracy at $(4.634, 10^{-9})$-DP.

The smaller datasets serve for quick experiments. **FEMNIST** gives EMNIST's handwritten characters back to their 3,550 writers, **Shakespeare** makes each speaking role a user ([LEAF](https://leaf.cmu.edu/)), and Apple's **FLAIR** has Flickr photos split by user ([FLAIR](https://github.com/apple/ml-flair)).

The deployed systems that these datasets simulate are not public. For example, Google trains its Gboard models on device with DP-FTRL. Xu et al. (2023) state the guarantee in zero-concentrated DP (zCDP), a form of the guarantee with a single parameter $\rho$ that converts to an $(\varepsilon, \delta)$ pair. The launched models have $\rho$ between 0.2 and 2. At a probability of exception $\delta = 10^{-10}$, that is an $\varepsilon$ between 4.49 and 13.69. Stack Overflow, with about 400 posts per user, is the public stand-in for that setting.

## What user-level data costs: Online Retail

Online Retail is the one dataset in the note's script with a user identifier, so it shows here what user-level privacy does to a release. Of its 541,909 invoice lines, 406,829 carry one of 4,372 customer ids, and one line is one product on one invoice. The left panel of @fig-contributions shows how many lines each identified customer has. The median customer has 42 lines, one in ten has more than 212, one in a hundred has more than 672, and the heaviest has 7,983. Any dataset where people act repeatedly has a long tail like this, and that tail makes user-level privacy expensive.

![Left: the share of Online Retail's customers with more than k invoice lines. Right: the error of a user-level DP histogram of lines per month, against the number of lines each customer is allowed to keep.](figures/contributions.svg){#fig-contributions}

Consider a release of the number of lines in each of the thirteen months, with the customer as the privacy unit. One customer added or removed moves all of their lines, so the histogram's sensitivity is the largest number of lines of any customer, 7,983, and Laplace noise of that scale makes the release useless. The standard fix is to **bound the contribution**: choose a bound $k$ and keep at most $k$ lines of each customer, here the first $k$ in date order. That caps the sensitivity at $k$, but it drops the rest of a heavy customer's lines, so the release has two errors that move in opposite directions. The dropped lines make every month's count too low, and that bias shrinks as $k$ grows. The Laplace noise has scale $k/\varepsilon$, so it grows with $k$. The right panel of @fig-contributions plots the root mean squared error over the thirteen months against $k$, and @tbl-contributions gives some of its points.

{{< include figures/_contributions.md >}}

At $\varepsilon = 1$ the error is smallest at a bound of 1,263 lines, where it is 3,294 lines against a monthly average of 31,295, about a tenth of the count. If you keep every line, the error is 11,290, more than three times worse. A bound that is too tight is also worse. At $k = 42$, the median, the bias alone is 21,146 lines a month. At $\varepsilon = 0.1$ the best bound moves down to 399 lines, with an error of 8,421. The best bound and the cost of privacy thus depend only on the tail of the per-user counts, so a user-level experiment must report those counts.

# Locations, graphs and clinical records {#sec-other}

Three kinds of data have DP literatures and datasets of their own. Locations are the most sensitive data that most people generate, graphs make the privacy unit a choice, and clinical records are the data DP was meant for but that you can rarely download.

## Locations and trajectories

A trajectory is a sequence of timed positions, and one person's trajectory is one record, so trajectory datasets are user-level by construction. **GeoLife** holds 17,621 trajectories of 182 volunteers in Beijing ([GeoLife](https://www.microsoft.com/en-us/download/details.aspx?id=52367)), and **T-Drive**, from the same Microsoft group, holds one week of 10,357 Beijing taxis. The **Porto taxi** data hold 1,710,671 trips of 442 taxis over one year, under CC BY 4.0 at UCI ([Porto](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015)), and most DP trajectory papers use them. Two check-in datasets are also in use, **Gowalla** from the Stanford Network Analysis Project (SNAP), with user ids and a friendship graph, and the **Foursquare** check-ins of New York and Tokyo.

Most DP work on these datasets is synthesis. AdaTrace, by Gursoy et al. (2018), and PrivTrace, by Wang et al. (2022), generate synthetic trajectories on GeoLife and Porto. Buchholz et al. (2024) review six generative models on GeoLife and Foursquare and find that most give no formal guarantee. Earlier, Qardaji, Yang and Li (2013) released two-dimensional counts over Gowalla check-ins, and the range-query benchmark DPBench of Hay et al. (2016) kept Gowalla among its datasets.

## Graphs

In a graph the privacy unit is a choice. **Edge-level** DP hides one relationship, and **node-level** DP hides a person with all of their relationships. The second is far harder, because one node can touch many edges, so a count can change by the person's whole degree. Most graph datasets come from SNAP, which states no licence for its collections. Karwa et al. (2011) released subgraph counts of arXiv collaboration graphs and of the Enron email graph, and the **ego-Facebook** graph of 4,039 users and 88,234 friendships is the small standard. Under local DP, where each person randomizes their own data before it leaves them, Imola, Murakami and Chaudhuri (2021) counted triangles and stars on graphs of up to three million nodes. Papers on DP graph neural networks use the citation graph **Cora** of 2,708 papers and larger graphs of Reddit, Amazon and the Open Graph Benchmark. Sajadmanesh et al. (2023) and Daigavane et al. (2021) are two examples.

## Clinical records

**MIMIC-III** is the dataset of DP clinical papers ([PhysioNet](https://physionet.org/content/mimiciii/1.4/)). It holds the intensive-care unit (ICU) records of over forty thousand patients at one Boston hospital, from 2001 to 2012. Access needs PhysioNet credentials, a training course and a signed data use agreement, and the agreement forbids you to share anything derived from the data. So the DP-GAN of Xie et al. (2018) and the DP-auto-GAN of Tantipongpipat et al. (2019) train on it, but nobody can release their synthetic records. Suriyakumar et al. (2021) show on the same data that DP-SGD hurts the rarest patients most.

The other clinical databases follow the same pattern. **MIMIC-IV** and the multi-hospital **eICU** database have the same terms as MIMIC-III. The two clinical datasets that you can download freely are the PhysioNet 2012 challenge, with 12,000 ICU stays, and Diabetes 130-US hospitals from @sec-tabular. The **Texas Hospital Inpatient Discharge** files are in between: you can download them, but under a state agreement with penalties for re-identification. Stadler, Oprisanu and Troncoso (2022) used samples of them to attack synthetic data.

# Public DP releases as datasets {#sec-released}

A few organizations have published data with a DP guarantee, and these releases are datasets too, each with a known $\varepsilon$. They show what budgets are used in practice, and they are the only public data whose noise a reader can study. @tbl-released lists them.

| Release | What is published | Guarantee | Where |
|:---|:---|:---|:---|
| US Census 2020 redistricting data | population counts by geography | $\varepsilon = 19.61$ in total, zCDP accounting | [data.census.gov](https://data.census.gov) |
| 2010 Census demonstration data | the same algorithm on 2010 data, and the noisy measurement file | $\varepsilon = 4.5$ and 12.2 (2021), production settings (2023) | [Census Bureau](https://www.census.gov/programs-surveys/decennial-census/decade/2020/planning-management/process/disclosure-avoidance/2020-das-development.html) |
| Google Community Mobility Reports | daily change in visits by place category and region, 2020–2022 | Laplace, $\varepsilon = 0.11$ to 0.44 per metric per day, 1.76 per user per day | [Google](https://www.google.com/covid19/mobility/) |
| Wikimedia pageviews by country | daily page views per country and project | $\rho$-zCDP of 0.015 per device per day, about $\varepsilon = 1$ at $\delta = 10^{-7}$ | [Wikimedia](https://analytics.wikimedia.org/published/datasets/country_project_page/00_README.html) |
| Google Community Mobility Reports | daily change in visits by place category and region, 2020–2022 | Laplace, $\varepsilon = 0.11$ to 0.44 per metric per day, 1.76 per user per day | [Google](https://www.google.com/covid19/mobility/) |
| Wikimedia pageviews by country | daily page views per country and project | $\rho$-zCDP of 0.015 per device per day, about $\varepsilon = 1$ at $\delta = 10^{-7}$ | [Wikimedia](https://analytics.wikimedia.org/published/datasets/country_project_page/00_README.html) |
| Meta Full URLs dataset | shares and views of 38 million URLs | Gaussian noise; action-level privacy, which protects one share or view rather than a user | by application to Social Science One |
| LinkedIn Audience Engagements API | top-k engagement counts | $(34.9, 7 \times 10^{-9})$-DP per analyst per month | not public |
| Apple, Microsoft telemetry | emoji and domain frequencies; app usage | local DP, $\varepsilon = 4$ (Apple), $\varepsilon = 1$ per collection (Microsoft) | not public |

: Real DP releases and their stated guarantees, as the publishing organization reports them. {#tbl-released}

Google's reports are the most useful of these to study, because Aktay et al. (2020) describe them exactly. Each metric is a daily percentage change in visits to a place category, with Laplace noise of $\varepsilon$ between 0.11 and 0.44 per metric per day. All of one user's contributions in a day share a budget of $\varepsilon = 1.76$. The reports stopped in October 2022, but you can still download the history, with its noise in it. Wikimedia's release of pageviews by country, described by Adeleye et al. (2023), is the other continuing release. Apple's mobility reports of the same period are not DP, though lists often put them next to Google's. Apple, Microsoft and LinkedIn publish the mechanisms and budgets of their deployments, but not their data.

# Datasets for DP synthetic data generation {#sec-synthetic}

A DP synthetic dataset is the output of a mechanism that looks like the original data: it has the same columns and the same kinds of rows, and no one record changed it much. The field needs its own benchmarks, because "looks like" has to be measured. This section first says why the methods that win measure marginals, then lists the challenges, the benchmark suites and the generators with their data.

## Why the winning generators fit marginals: Adult's domain

The winning generators all measure a few **marginals**, tables of counts over a few attributes, and fit a model to the noisy counts. The reason is the size of a table's domain, and the note *AIM, Explained From Scratch* works it out in full on Adult. @tbl-adult-domain gives the number of values of each of Adult's fifteen columns, with each numeric attribute cut into ten equal-width bins. The full table of counts over all fifteen columns has one cell per combination of values. That is about $7.6 \times 10^{13}$ cells for 48,842 records. So almost every true count is 0 or 1. Without noise, the table lists the records one by one. With Laplace noise of standard deviation $\sqrt{2}/\varepsilon$ per cell, the noise buries every count.

{{< include figures/_adult_domain.md >}}

Small marginals are at the other end, as @tbl-adult-marginals shows. When basic composition splits a budget of $\varepsilon = 1$ evenly over all 105 two-way marginals, each cell gets noise with standard deviation 148, against an average count of 430, which is a usable signal. Over the 455 three-way marginals the noise is 644 per cell, more than the average count of 43. So a practical generator measures all the one-way marginals and a chosen set of two- and three-way ones, and fits a model to the noisy measurements.

## The NIST challenges

The public record of the field is three NIST challenges, each with public data and published scores. The **2018 Differential Privacy Synthetic Data Challenge** ran three matches ([NIST](https://www.nist.gov/ctl/pscr/open-innovation-prize-challenges/past-prize-challenges/2018-differential-privacy-synthetic)). The first two used San Francisco Fire Department calls for service, about 305,000 records with 32 attributes. The third used the 1940 Census PUMS for Colorado, about 661,000 records with 98 attributes. Bowen and Snoke (2021) compare the entries.

The results set the pattern for everything since. Ryan McKenna's entry won the third match and the overall ranking, and the winning methods measured low-order marginals with noise and fitted a graphical model. The one entry built on a GAN scored far below them. McKenna, Miklau and Sheldon (2021) turned the winner into **MST**. It measures all one-way marginals, selects two-way marginals along a maximum spanning tree of mutual information, and fits a graphical model to the noisy measurements with Private-PGM, described with @tbl-generators.

The **2020 Differential Privacy Temporal Map Challenge** moved to data with time and place, from Baltimore 911 calls, the ACS and Chicago taxi trips ([NIST](https://www.nist.gov/ctl/pscr/open-innovation-prize-challenges/past-prize-challenges/2020-differential-privacy-temporal)). Noisy marginals and graphical models won again. The winning code is public, and McKenna's entry became the basis of **AIM** ([nist-synthetic-data-2021](https://github.com/ryan112358/nist-synthetic-data-2021)).

The **Collaborative Research Cycle**, from 2023, is a shared benchmark, not a competition ([NIST CRC](https://pages.nist.gov/privacy_collaborative_research_cycle/)). Its data are the **NIST ACS Data Excerpts**, drawn from the 2019 ACS PUMS with 24 features. There is a national excerpt of 27,254 records and smaller ones for Massachusetts and Texas, each with a control sample from 2018 ([SDNist](https://github.com/usnistgov/SDNist)). As Sen et al. (2023) describe, the areas were chosen for their demographic diversity, so that a generator's errors on minorities show. The SDNist tool scores any synthetic version against the originals, and the archive holds over 450 scored submissions, the largest public comparison of DP synthetic data.

## Benchmark suites and what they found

**SDGym**, from the CTGAN paper of Xu et al. (2019), fixed the datasets of the deep-learning side ([SDGym](https://github.com/sdv-dev/SDGym)). Its real datasets include Adult, Census-Income KDD, Credit Card Fraud and two binarized MNISTs, and it adds seven simulated datasets. CTGAN itself is not private, but DP-CTGAN, PATE-CTGAN and DP-MERF were evaluated on SDGym's datasets.

Tao, McKenna, Hay, Machanavajjhala and Miklau (2021) ran the first systematic DP comparison, on seven small tables that include Adult, Bank Marketing and Census-Income KDD. They found that many mechanisms, and nearly all the GAN-based ones, did not keep even the one-way marginals. No mechanism was best everywhere, but MST had the best average rank, and the bins of the numeric columns mattered as much as the mechanism. Ganev, Xu and De Cristofaro (2024) trained about 21,000 models to explain why. Graphical models spread their budget over columns and struggle on wide tables, while deep models spend it per training step and scale to width. So PATE-GAN wins on complex prediction tasks and MST on statistics, and Chen et al. (2025) find the same trade-off on ACS, Bank and Loan data.

The benchmarks also showed a problem of reproducibility. Ganev, Annamalai and De Cristofaro (2025) tried six open-source implementations of PATE-GAN. None matched the paper's utility, and all leaked more privacy than claimed. So a result on a benchmark is only as good as the audit of its implementation.

## The generators and their data

@tbl-generators lists the main generators with the data of their evaluations, which you need to compare a new method with them.

| Generator | Kind | Evaluated on | Source |
|:---|:---|:---|:---|
| PrivBayes (2017) | Bayesian network over marginals | NLTCS 21,574; ACS 47,461; Adult 45,222; Brazil 2000 census 38,000 | Zhang et al. |
| MST (2021) | marginals and Private-PGM | the 2018 NIST data | McKenna, Miklau and Sheldon |
| AIM (2022) | adaptive marginals and Private-PGM | Adult; Salary 135,727; MSNBC 989,818; Fire 305,119; NLTCS; Titanic 1,304 | McKenna et al. |
| PrivSyn (2021) | many two-way marginals, iterative fitting | Adult; US Accidents 600,000; Loan 600,000; the Colorado 1940 PUMS | Zhang et al. |
| DP-GAN (2018) | GAN with DP-SGD | MNIST; MIMIC-III binary codes | Xie et al. |
| PATE-GAN (2019) | GAN trained through PATE | Credit Card Fraud; cervical cancer; ISOLET; epileptic seizure; two private | Jordon, Yoon and van der Schaar |
| DP-MERF (2021) | kernel mean embeddings | MNIST; Fashion-MNIST; nine SDGym tables | Harder, Adamczewski and Park |
| DP diffusion (2022–23) | diffusion models with DP-SGD | MNIST; Fashion-MNIST; CelebA; CIFAR-10; Camelyon17 | Dockhorn et al.; Ghalebikesabi et al. |
| Private Evolution (2024) | API calls to a foundation model | CIFAR-10; Camelyon17; two sets of 100 cat photos | Lin et al. |
| Aug-PE (2024) | the same for text | Yelp 1.9 million; OpenReview 8,396; PubMed 75,316 | Xie et al. |

: DP synthetic data generators and the datasets of their evaluations, as their papers report them. {#tbl-generators}

The marginal-based line began with **PrivBayes** of Zhang et al. (2017), which learns a Bayesian network under DP and samples from it. **Private-PGM** of McKenna, Sheldon and Miklau (2019) then separated measurement from fit: it fits a graphical model to any set of noisy marginals, whatever mechanism measured them. MST chooses those marginals along a spanning tree, and **AIM** of McKenna et al. (2022) chooses them adaptively, by how badly the current model fits them. The reference implementation of these methods now ships only Adult and its domain file ([mbi](https://github.com/ryan112358/mbi)), so you must fetch every other dataset of @tbl-generators from its source.

The GAN line runs from **DP-GAN** through **PATE-GAN** to **DP-MERF**, with evaluations on SDGym's tables and on MNIST. The systematization of Hu et al. (2024) finds DP-MERF the most dependable of the deep methods on images.

The newest line does not train on the private data at all. **Private Evolution**, by Lin et al. (2024), asks a pre-trained image model, through its application programming interface (API), for candidate images. It scores the candidates by DP nearest-neighbour counts against the private data, keeps the best and repeats. On CIFAR-10 it reaches an FID of 7.87 at $\varepsilon = 0.67$, where earlier methods needed $\varepsilon = 32$.

**Aug-PE**, by Xie et al. (2024), does the same for text, on Yelp, OpenReview and PubMed data that the authors prepared ([aug-pe](https://github.com/AI-secure/aug-pe)). The Yelp data come from Yue et al. (2023), who first showed that DP fine-tuning of GPT-2 gives synthetic reviews on which a classifier trains almost as well as on the originals. Only the code in the repository has a licence, so check the terms of the texts before you use them again.

## Evaluating synthetic data, and attacking it

A synthetic dataset is judged on utility and on privacy, and both need the original. To measure utility, compare the marginals, or train a model on the synthetic data and test it on real held-out data, as SDNist and the SDMetrics library do ([SDMetrics](https://github.com/sdv-dev/SDMetrics)). Privacy is measured by attack, and here the datasets matter because the attacks target specific records. For example, Stadler, Oprisanu and Troncoso (2022) attacked synthetic versions of Adult and of the Texas discharge data. Without DP, the synthetic data did not stop linkage and attribute inference. With DP the synthesis protected the records, but at a large cost in utility.

Annamalai, Gadotti and Rocher (2024) then built a linear reconstruction attack that infers a hidden attribute from DP synthetic data, on the ACS California employment task and the Fire data, and found that it works on ordinary records, not only on outliers. So the best datasets for attacks have a known sensitive attribute and a known population, which the ACS tasks provide.

Some official synthetic products, such as the Census Bureau's SIPP Synthetic Beta, often appear next to DP ones but are not DP. The UK Office for National Statistics, by contrast, used MST on the linked 2011 Census and deaths register for England and Wales, about 55 million rows. Its DP synthetic version has $\varepsilon = 1$ and a probability of exception $\delta = 10^{-8}$ ([ONS Data Science Campus](https://datasciencecampus.ons.gov.uk/synthesising-the-linked-2011-census-and-deaths-dataset-while-preserving-its-confidentiality/)).

# Choosing a dataset for a privacy-utility study {#sec-choose}

A privacy-utility study prices one decision: it trains or releases at several values of $\varepsilon$ and asks how much of a business result survives at each. This section says what a dataset needs for such a study, and which datasets above would do. First, the study needs a business count, a label or outcome whose count the decision maker cares about, so an unlabelled dataset will not do. Second, the records must be many enough for the curve of utility against $\varepsilon$ to have a flat part, which @sec-what put at tens of thousands at least. Third, a public codebook lets you fix the ranges without a look at the private data. Fourth, the licence must allow you to publish the results, and ideally the data. @tbl-choose rates the candidates.

| Dataset | The business count | Records | Fits a privacy-utility study because | But |
|:---|:---|---:|:---|:---|
| Adult | high earners reached by a campaign | 48,842 | codebook, licence, every DP method has a result on it | 1994 data, retiring |
| ACSIncome, one state | the same, on current data | thousands to hundreds of thousands | same task, any size, same source as the Census Bureau's DP work | no codebook of categories bundled; use the PUMS data dictionary |
| Bank Marketing | deposits sold per contact | 45,211 | a real campaign with a yes-or-no outcome | two versions; contacts repeat clients |
| Credit Card Fraud | frauds caught per review | 284,807 | the count that matters is rare, so DP's cost shows | features are anonymized components with no codebook |
| Online Retail | revenue retained per customer | 4,372 customers | the only free table for a user-level study | no label; the outcome must be defined |
| Home Credit | defaults avoided per loan | 307,511 | a real credit decision with rich features | competition terms, unverified |
| Stack Overflow | next-word accuracy | 342,477 users | the standard user-level benchmark | not a business count |

: Candidate datasets for a privacy-utility study, with the count a decision maker would sign for. {#tbl-choose}

Three of these need a comment. ACSIncome for one state is the natural replacement for a study on Adult. It is the same task on data twenty-five years newer, and you can choose a state that matches the size of the business. Credit Card Fraud makes the study harder, because its business count is tiny, 492 frauds among 284,807 transactions. So a model that must catch a fixed number of frauds with a fixed number of reviews feels DP's noise, where a model on Adult would not. Online Retail, with its customer ids, is the only free dataset on which the privacy unit can be a customer. A study of user-level DP for a business would start there, and its first decision would be the contribution bound of @sec-users, the most lines each customer may keep.

Whatever the dataset, the report should say four things about it, which Ponomareva et al. (2023) put in their checklist. It says what one record is, and so whom the guarantee covers. It says which neighbouring relation the accounting assumed, add/remove or replace-one. It says what public data the method used, for pre-training, for tuning or for the codebook. And for a user-level study it gives the distribution of records per user, because without it no reader can judge the contribution bound.

# References {#sec-sources}

Datasets, in the order of the catalogue.

- Becker, B. and Kohavi, R. (1996). [Adult](https://archive.ics.uci.edu/dataset/2/adult). UCI Machine Learning Repository, dataset 2. CC BY 4.0.
- Ding, F., Hardt, M., Miller, J. and Schmidt, L. (2021). [Retiring Adult: New Datasets for Fair Machine Learning](https://arxiv.org/abs/2108.04884). *NeurIPS 2021*. The folktables library and the ACS tasks.
- [Census-Income (KDD)](https://archive.ics.uci.edu/dataset/117/census+income+kdd), UCI dataset 117; Moro, S., Rita, P. and Cortez, P. (2014), [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), UCI dataset 222; Hofmann, H. (1994), [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), UCI dataset 144; Chen, D. (2015), [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), UCI dataset 352; Clore, J. et al. (2014), [Diabetes 130-US hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), UCI dataset 296. All CC BY 4.0.
- Dal Pozzolo, A. et al. [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Kaggle, Université Libre de Bruxelles Machine Learning Group. DbCL 1.0.
- Harper, F. M. and Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* 5(4). [GroupLens](https://files.grouplens.org/datasets/movielens/).
- Narayanan, A. and Shmatikov, V. (2008). [Robust De-anonymization of Large Sparse Datasets](https://arxiv.org/abs/cs/0610105). *IEEE S&P 2008*. The Netflix Prize de-anonymization.
- McSherry, F. and Mironov, I. (2009). Differentially Private Recommender Systems: Building Privacy into the Netflix Prize Contenders. *KDD 2009*.
- Jain, P., Thakkar, O. and Thakurta, A. (2018). [Differentially Private Matrix Completion Revisited](https://arxiv.org/abs/1712.09765). *ICML 2018*.
- Abowd, J. et al. (2022). [The 2020 Census Disclosure Avoidance System TopDown Algorithm](https://arxiv.org/abs/2204.08986). *Harvard Data Science Review*, special issue 2. The Census Bureau's [key parameters](https://www.census.gov/newsroom/press-releases/2021/2020-census-key-parameters.html) and [2010 demonstration data](https://www.census.gov/programs-surveys/decennial-census/decade/2020/planning-management/process/disclosure-avoidance/2020-das-development.html).
- Kenny, C. et al. (2021). [The use of differential privacy for census data and its impact on redistricting](https://arxiv.org/abs/2105.14197). *Science Advances* 7(41).
- Image datasets: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (MIT); Krizhevsky, A. (2009), [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html); [ImageNet](https://www.image-net.org/download.php); [Places365](http://places2.csail.mit.edu/download-private.html); Liu, Z. et al. (2015), [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html); Koh, P. W. et al. (2021), [WILDS](https://arxiv.org/abs/2012.07421), for Camelyon17.
- Abadi, M. et al. (2016). [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133). *ACM CCS 2016*. DP-SGD, with its MNIST and CIFAR-10 results.
- Tramèr, F. and Boneh, D. (2021). [Differentially Private Learning Needs Better Features (or Much More Data)](https://arxiv.org/abs/2011.11660). *ICLR 2021*.
- De, S., Berrada, L., Hayes, J., Smith, S. L. and Balle, B. (2022). [Unlocking High-Accuracy Differentially Private Image Classification through Scale](https://arxiv.org/abs/2204.13650).
- Kurakin, A. et al. (2022). [Toward Training at ImageNet Scale with Differential Privacy](https://arxiv.org/abs/2201.12328).
- Wei, C. et al. (2023). [DPMLBench: Holistic Evaluation of Differentially Private Machine Learning](https://arxiv.org/abs/2305.05900). *ACM CCS 2023*.
- Mokhtari, A., Kodeiri, S., Mohapatra, S., Tramèr, F. and Kamath, G. (2026). [Rethinking Benchmarks for Differentially Private Image Classification](https://arxiv.org/abs/2601.17189).
- Text datasets: [GLUE](https://www.tensorflow.org/datasets/catalog/glue); Novikova, J., Dušek, O. and Rieser, V. (2017), [E2E](https://github.com/tuetschek/e2e-dataset); Nan, L. et al. (2021), [DART](https://github.com/Yale-LILY/dart); Zhang, S. et al. (2018), [PersonaChat](https://arxiv.org/abs/1801.07243); Merity, S. et al. (2016), [WikiText](https://arxiv.org/abs/1609.07843); the [Enron corpus](https://www.cs.cmu.edu/~enron/); [C4](https://huggingface.co/datasets/allenai/c4); Gao, L. et al. (2020), [the Pile](https://pile.eleuther.ai/).
- Li, X., Tramèr, F., Liang, P. and Hashimoto, T. (2022). [Large Language Models Can Be Strong Differentially Private Learners](https://arxiv.org/abs/2110.05679). *ICLR 2022*.
- Yu, D. et al. (2022). [Differentially Private Fine-tuning of Language Models](https://arxiv.org/abs/2110.06500). *ICLR 2022*.
- Anil, R., Ghazi, B., Gupta, V., Kumar, R. and Manurangsi, P. (2022). [Large-Scale Differentially Private BERT](https://arxiv.org/abs/2108.01624).
- Tramèr, F., Kamath, G. and Carlini, N. (2024). [Position: Considerations for Differentially Private Learning with Large-Scale Public Pretraining](https://arxiv.org/abs/2212.06470). *ICML 2024*.
- Caldas, S. et al. (2018). [LEAF: A Benchmark for Federated Settings](https://arxiv.org/abs/1812.01097). The [TensorFlow Federated datasets](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets), including [Stack Overflow](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data).
- Song, C., Granqvist, F. and Talwar, K. (2022). [FLAIR: Federated Learning Annotated Image Repository](https://arxiv.org/abs/2207.08869). [GitHub](https://github.com/apple/ml-flair).
- McMahan, H. B., Ramage, D., Talwar, K. and Zhang, L. (2018). [Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1710.06963). *ICLR 2018*.
- Kairouz, P., McMahan, B., Song, S., Thakkar, O., Thakurta, A. and Xu, Z. (2021). [Practical and Private (Deep) Learning without Sampling or Shuffling](https://arxiv.org/abs/2103.00039). *ICML 2021*. DP-FTRL on Stack Overflow.
- Xu, Z. et al. (2023). [Federated Learning of Gboard Language Models with Differential Privacy](https://arxiv.org/abs/2305.18465). The deployed guarantees.
- Location data: Zheng, Y. et al., [GeoLife](https://www.microsoft.com/en-us/download/details.aspx?id=52367) and [T-Drive](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/); Moreira-Matias, L. et al. (2013), [Porto taxi trajectories](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015), UCI dataset 339; Cho, E., Myers, S. and Leskovec, J. (2011), [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html); Yang, D. et al., [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset).
- Gursoy, M. E., Liu, L., Truex, S., Yu, L. and Wei, W. (2018). Utility-Aware Synthesis of Differentially Private and Attack-Resilient Location Traces. *ACM CCS 2018*. AdaTrace.
- Wang, H. et al. (2022). [PrivTrace: Differentially Private Trajectory Synthesis by Adaptive Markov Models](https://arxiv.org/abs/2210.00581). *USENIX Security 2023*.
- Buchholz, E., Abuadbba, A., Wang, S., Nepal, S. and Kanhere, S. (2024). [SoK: Can Trajectory Generation Combine Privacy and Utility?](https://arxiv.org/abs/2403.07218). *PoPETs 2024*.
- Qardaji, W., Yang, W. and Li, N. (2013). [Differentially Private Grids for Geospatial Data](https://arxiv.org/abs/1209.1322). *ICDE 2013*.
- Aktay, A. et al. (2020). [Google COVID-19 Community Mobility Reports: Anonymization Process Description](https://arxiv.org/abs/2004.04145). The [reports](https://www.google.com/covid19/mobility/).
- Adeleye, T. et al. (2023). [Publishing Wikipedia Usage Data with Strong Privacy Guarantees](https://arxiv.org/abs/2308.16298). The [dataset](https://analytics.wikimedia.org/published/datasets/country_project_page/00_README.html).
- Rogers, R. et al. (2020). [LinkedIn's Audience Engagements API](https://arxiv.org/abs/2002.05839); Apple (2017), [Learning with Privacy at Scale](https://machinelearning.apple.com/research/learning-with-privacy-at-scale); Ding, B., Kulkarni, J. and Yekhanin, S. (2017), [Collecting Telemetry Data Privately](https://arxiv.org/abs/1712.01524), *NeurIPS 2017*; Messing, S. et al. (2020), [Facebook Privacy-Protected Full URLs Data Set](https://doi.org/10.7910/DVN/TDOAPG).
- Graph data: Leskovec, J. and Krevl, A., [SNAP Datasets](https://snap.stanford.edu/data/); Karwa, V., Raskhodnikova, S., Smith, A. and Yaroslavtsev, G. (2011), Private Analysis of Graph Structure, *PVLDB* 4(11); Imola, J., Murakami, T. and Chaudhuri, K. (2021), [Locally Differentially Private Analysis of Graph Statistics](https://arxiv.org/abs/2010.08688), *USENIX Security 2021*; Sajadmanesh, S., Shamsabadi, A. S., Bellet, A. and Gatica-Perez, D. (2023), [GAP: Differentially Private Graph Neural Networks with Aggregation Perturbation](https://arxiv.org/abs/2203.00949), *USENIX Security 2023*; Daigavane, A. et al. (2021), [Node-Level Differentially Private Graph Neural Networks](https://arxiv.org/abs/2111.15521).
- Clinical data: Johnson, A. et al. (2016), MIMIC-III, a freely accessible critical care database, *Scientific Data* 3, on [PhysioNet](https://physionet.org/content/mimiciii/1.4/); [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/); [eICU](https://physionet.org/content/eicu-crd/2.0/); the [PhysioNet 2012 challenge](https://physionet.org/content/challenge-2012/1.0.0/); the [Texas Hospital Inpatient Discharge PUDF](https://www.dshs.texas.gov/center-health-statistics/chs-data-sets-reports/texas-health-care-information-collection/general-public-information/hospital-discharge-data-public).
- Suriyakumar, V. M., Papernot, N., Goldenberg, A. and Ghassemi, M. (2021). [Chasing Your Long Tails: Differentially Private Prediction in Health Care Settings](https://arxiv.org/abs/2010.06667). *FAccT 2021*.
- Bowen, C. M. and Snoke, J. (2021). [Comparative Study of Differentially Private Synthetic Data Algorithms from the NIST PSCR Differential Privacy Synthetic Data Challenge](https://arxiv.org/abs/1911.12704). *Journal of Privacy and Confidentiality* 11(1).
- Sen, A., Task, C., Kapur, D., Howarth, G. and Bhagat, K. (2023). [Diverse Community Data for Benchmarking Data Privacy Algorithms](https://arxiv.org/abs/2306.13216). *NeurIPS 2023 Datasets and Benchmarks*. The NIST excerpts and [SDNist](https://github.com/usnistgov/SDNist).
- Xu, L., Skoularidou, M., Cuesta-Infante, A. and Veeramachaneni, K. (2019). [CTGAN, the conditional GAN for tabular data](https://arxiv.org/abs/1907.00503). *NeurIPS 2019*. With the [SDGym](https://github.com/sdv-dev/SDGym).
- Tao, Y., McKenna, R., Hay, M., Machanavajjhala, A. and Miklau, G. (2021). [Benchmarking Differentially Private Synthetic Data Generation Algorithms](https://arxiv.org/abs/2112.09238).
- Ganev, G., Xu, K. and De Cristofaro, E. (2024). [Graphical vs. Deep Generative Models: Measuring the Impact of Differentially Private Mechanisms and Budgets on Utility](https://arxiv.org/abs/2305.10994). *ACM CCS 2024*.
- Ganev, G., Annamalai, M. S. M. S. and De Cristofaro, E. (2025). [The Elusive Pursuit of Reproducing PATE-GAN: Benchmarking, Auditing, Debugging](https://arxiv.org/abs/2406.13985). *TMLR*.
- Chen, K., Li, X., Gong, C., McKenna, R. and Wang, T. (2025). [Benchmarking Differentially Private Tabular Data Synthesis](https://arxiv.org/abs/2504.14061).
- Hu, Y. et al. (2024). [SoK: Privacy-Preserving Data Synthesis](https://arxiv.org/abs/2307.02106). *IEEE S&P 2024*.
- Zhang, J., Cormode, G., Procopiuc, C. M., Srivastava, D. and Xiao, X. (2017). PrivBayes: Private Data Release via Bayesian Networks. *ACM Transactions on Database Systems* 42(4).
- McKenna, R., Sheldon, D. and Miklau, G. (2019). [Graphical-model based estimation and inference for differential privacy](https://arxiv.org/abs/1901.09136). *ICML 2019*. Private-PGM, in the [mbi](https://github.com/ryan112358/mbi) repository.
- McKenna, R., Miklau, G. and Sheldon, D. (2021). [Winning the NIST Contest: A scalable and general approach to differentially private synthetic data](https://arxiv.org/abs/2108.04978). *Journal of Privacy and Confidentiality* 11(3). MST.
- McKenna, R., Mullins, B., Sheldon, D. and Miklau, G. (2022). [AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data](https://arxiv.org/abs/2201.12677). *PVLDB* 15(11).
- Zhang, Z. et al. (2021). [PrivSyn: Differentially Private Data Synthesis](https://arxiv.org/abs/2012.15128). *USENIX Security 2021*.
- Xie, L., Lin, K., Wang, S., Wang, F. and Zhou, J. (2018). [Differentially Private Generative Adversarial Network](https://arxiv.org/abs/1802.06739).
- Jordon, J., Yoon, J. and van der Schaar, M. (2019). [PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees](https://openreview.net/forum?id=S1zk9iRqF7). *ICLR 2019*.
- Harder, F., Adamczewski, K. and Park, M. (2021). [DP-MERF](https://arxiv.org/abs/2002.11603). *AISTATS 2021*.
- Dockhorn, T., Cao, T., Vahdat, A. and Kreis, K. (2022). [Differentially Private Diffusion Models](https://arxiv.org/abs/2210.09929). *TMLR*.
- Ghalebikesabi, S. et al. (2023). [Differentially Private Diffusion Models Generate Useful Synthetic Images](https://arxiv.org/abs/2302.13861).
- Lin, Z., Gopi, S., Kulkarni, J., Nori, H. and Yekhanin, S. (2024). [Differentially Private Synthetic Data via Foundation Model APIs 1: Images](https://arxiv.org/abs/2305.15560). *ICLR 2024*. Private Evolution, in the [DPSDA](https://github.com/microsoft/DPSDA) repository.
- Xie, C. et al. (2024). [Differentially Private Synthetic Data via Foundation Model APIs 2: Text](https://arxiv.org/abs/2403.01749). *ICML 2024*. Aug-PE.
- Yue, X. et al. (2023). [Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe](https://arxiv.org/abs/2210.14348). *ACL 2023*.
- Stadler, T., Oprisanu, B. and Troncoso, C. (2022). [Synthetic Data – Anonymisation Groundhog Day](https://arxiv.org/abs/2011.07018). *USENIX Security 2022*.
- Annamalai, M. S. M. S., Gadotti, A. and Rocher, L. (2024). [A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data](https://arxiv.org/abs/2301.10053). *USENIX Security 2024*.
- Hardt, M., Ligett, K. and McSherry, F. (2012). [A Simple and Practical Algorithm for Differentially Private Data Release](https://arxiv.org/abs/1012.4763). *NeurIPS 2012*. The multiplicative weights exponential mechanism.
- Aydore, S. et al. (2021). [Differentially Private Query Release Through Adaptive Projection](https://arxiv.org/abs/2103.06641). *ICML 2021*. RAP, on Adult and Loans; Vietri, G. et al. (2020), [New Oracle-Efficient Algorithms for Private Synthetic Data Release](https://arxiv.org/abs/2007.05453), *ICML 2020*, for the Loans data; Vietri, G. et al. (2022), [Private Synthetic Data for Multitask Learning and Marginal Queries](https://arxiv.org/abs/2209.07400), *NeurIPS 2022*, for the ACS benchmark.
- Hay, M., Machanavajjhala, A., Miklau, G., Chen, Y. and Zhang, D. (2016). [Principled Evaluation of Differentially Private Algorithms using DPBench](https://arxiv.org/abs/1512.04817). *SIGMOD 2016*. Twenty-seven datasets for range queries.
- Ponomareva, N. et al. (2023). [How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy](https://arxiv.org/abs/2303.00654). *Journal of Artificial Intelligence Research* 77. The reporting checklist.
- Office for National Statistics Data Science Campus (2023). [Synthesising the linked 2011 Census and deaths dataset while preserving its confidentiality](https://datasciencecampus.ons.gov.uk/synthesising-the-linked-2011-census-and-deaths-dataset-while-preserving-its-confidentiality/).
